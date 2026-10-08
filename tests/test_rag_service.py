"""RAG Service 测试：验证低相关度 Query Rewrite 的最小闭环。"""

from __future__ import annotations

from db.vector.store import ChunkHit
from service.rag.rag_service import RAGService


class RecordingStore:
    """不调用真实 Embedding 的最小测试向量库。"""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 3) -> list[ChunkHit]:
        self.queries.append(query)
        score = 0.78 if "售后" in query else 0.08
        return [
            ChunkHit(
                chunk_id="refund-policy-1",
                title="退款政策",
                source="refund_policy.md",
                section="退款条件",
                content="商品存在质量问题时可以申请售后。",
                score=score,
            )
        ]


def test_low_relevance_query_is_rewritten_and_retried() -> None:
    store = RecordingStore()
    result = RAGService(store=store).search("手机坏了怎么办")

    assert store.queries == ["手机坏了怎么办", "手机质量问题 售后怎么办"]
    assert result.rewrite_applied is True
    assert result.rewrite_method == "rule"
    assert result.retrieval_query == "手机质量问题 售后怎么办"
    assert result.hits[0].score == 0.78


def test_rewrite_is_not_used_when_direct_hit_is_relevant() -> None:
    store = RecordingStore()
    result = RAGService(store=store).search("售后政策")

    assert store.queries == ["售后政策"]
    assert result.rewrite_applied is False
    assert result.retrieval_query == "售后政策"


def test_empty_query_does_not_touch_vector_store() -> None:
    store = RecordingStore()
    result = RAGService(store=store).search("  ")

    assert store.queries == []
    assert result.hits == []


class LLMRewriteStore:
    """只对测试指定的改写词返回高分，不依赖真实模型或 Embedding API。"""

    def __init__(self, improved_query: str = "规范化的售后政策查询") -> None:
        self.queries: list[str] = []
        self.improved_query = improved_query

    def search(self, query: str, top_k: int = 3) -> list[ChunkHit]:
        self.queries.append(query)
        score = 0.82 if query == self.improved_query else 0.12
        return [
            ChunkHit(
                chunk_id="after-sale-policy-1",
                title="售后政策",
                source="after_sale_policy.md",
                section="质量问题",
                content="商品质量问题可按售后政策处理。",
                score=score,
            )
        ]


def test_llm_rewrite_runs_only_after_rule_candidate_does_not_improve() -> None:
    store = LLMRewriteStore()
    rewritten_queries: list[str] = []

    def rewrite(query: str) -> str:
        rewritten_queries.append(query)
        return "规范化的售后政策查询"

    result = RAGService(store=store, query_rewriter=rewrite).search("这玩意儿坏了该咋整")

    assert rewritten_queries == ["这玩意儿坏了该咋整"]
    assert store.queries == [
        "这玩意儿坏了该咋整",
        "这玩意儿质量问题 售后该咋整",
        "规范化的售后政策查询",
    ]
    assert result.rewrite_applied is True
    assert result.rewrite_method == "llm"
    assert result.retrieval_query == "规范化的售后政策查询"
    assert result.hits[0].score == 0.82


def test_llm_rewrite_failure_keeps_original_result() -> None:
    store = LLMRewriteStore()

    def fail_rewrite(query: str) -> str:
        raise TimeoutError("model timeout")

    result = RAGService(store=store, query_rewriter=fail_rewrite).search("完全未覆盖的表达")

    assert result.rewrite_applied is False
    assert result.rewrite_method == ""
    assert result.retrieval_query == "完全未覆盖的表达"
    assert result.hits[0].score == 0.12


def test_llm_rewrite_candidate_is_not_used_when_it_does_not_improve() -> None:
    store = LLMRewriteStore(improved_query="never-used")
    result = RAGService(
        store=store,
        query_rewriter=lambda query: "another query",
    ).search("完全未覆盖的表达")

    assert result.rewrite_applied is False
    assert result.retrieval_query == "完全未覆盖的表达"
    assert result.hits[0].score == 0.12


def test_relevant_direct_hit_does_not_call_llm_rewriter() -> None:
    store = RecordingStore()
    rewrite_calls: list[str] = []

    result = RAGService(
        store=store,
        query_rewriter=lambda query: rewrite_calls.append(query) or "rewritten",
    ).search("售后政策")

    assert rewrite_calls == []
    assert result.rewrite_method == ""
    assert store.queries == ["售后政策"]
