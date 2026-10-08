"""知识库检索 Tool（RAG 入口）。"""

from __future__ import annotations

from langchain_core.tools import tool

from service.rag.rag_service import RAGService


@tool
def search_knowledge_base(query: str) -> dict:
    """检索企业知识库（政策、规则、使用说明等），返回最相关的知识片段。

    当原始问题相关度不足时，RAG Service 先试领域规则；规则未改善时再调用 LLM
    生成候选改写，并通过二次检索比较结果。Tool 不直接实现改写算法。

    Args:
        query: 检索问题或关键词
    """
    service = RAGService()
    result = service.search(query, top_k=3)
    return {"success": True, **result.to_dict()}
