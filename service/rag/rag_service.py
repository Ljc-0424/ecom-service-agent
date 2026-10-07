"""RAG 服务：企业知识检索 Pipeline。

第一版 Pipeline：
    Query → Retriever → Vector Store（轻量词向量）→ Result Formatting

Agent 对外只看到 search_knowledge_base Tool。
后续可替换 Query Rewrite / Rerank / Citation，不必改 Tool 接口。

【谁在用】
  agent/tools/knowledge_tools.py 的 search_knowledge_base：
  LLM 需要政策/规则类知识时调用它，它再调本文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from db.vector.store import ChunkHit, VectorStore


@dataclass
class RAGHit:
    """一条检索命中：知识片段 + 来源信息（给 LLM 看和引用的格式）。"""

    title: str  # 所属文档标题
    source: str  # 来源文件名（检索结果展示「出处」用）
    section: str  # 所属小节标题
    content: str  # 知识片段正文
    score: float  # 相似度得分（越大越相关）

    def to_dict(self) -> dict[str, Any]:
        """转成字典。"""
        return {
            "title": self.title,
            "source": self.source,
            "section": self.section,
            "content": self.content,
            # round(x, 4)：保留 4 位小数，纯粹为了日志/返回结果可读
            "score": round(self.score, 4),
        }


@dataclass
class RAGResult:
    """一次检索的整体结果：用户的问题 + 命中列表。"""

    query: str  # 用户检索问题原话
    hits: list[RAGHit] = field(default_factory=list)  # 命中的知识片段列表

    def to_dict(self) -> dict[str, Any]:
        """转成字典。"""
        return {
            "query": self.query,
            "hits": [h.to_dict() for h in self.hits],
            "hit_count": len(self.hits),
        }


class RAGService:
    """知识检索服务（对 VectorStore 的一层薄封装）。"""

    def __init__(self, store: Optional[VectorStore] = None) -> None:
        """初始化：默认用全局向量库单例，测试时可注入独立实例。"""
        self.store = store or VectorStore.get_default()

    def search(self, query: str, top_k: int = 3) -> RAGResult:
        """执行 RAG 检索并返回带来源信息的结果。

        【流程】
          1. 去掉首尾空白；空问题直接返回空结果（不浪费检索）
          2. 调 VectorStore.search 拿原始命中（向量相似度 Top-K）
          3. 转成 RAGHit 列表包装返回
        """
        query = (query or "").strip()
        if not query:
            return RAGResult(query=query, hits=[])
        raw_hits: list[ChunkHit] = self.store.search(query, top_k=top_k)
        return RAGResult(
            query=query,
            hits=[
                RAGHit(
                    title=h.title,
                    source=h.source,
                    section=h.section,
                    content=h.content,
                    score=h.score,
                )
                for h in raw_hits
            ],
        )
