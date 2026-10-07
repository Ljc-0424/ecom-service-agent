"""知识库 Tool（RAG 入口）。

【官方对照】@tool + Args docstring，同 order_tools.py 文件头说明。
【名字拆解】search_knowledge_base = search（搜索）+ knowledge base（知识库）。
  用 search 而不是 get：按内容模糊找，不是按编号精确取。
"""

from __future__ import annotations

from langchain_core.tools import tool

from service.rag.rag_service import RAGService


@tool
def search_knowledge_base(query: str) -> dict:
    """检索企业知识库（政策、规则、使用说明等），返回最相关的知识片段。

    Args:
        query: 检索问题或关键词
    """
    service = RAGService()
    result = service.search(query, top_k=3)
    return {"success": True, **result.to_dict()}
    # ** 把字典里的键值「摊开」合并进外层字典
