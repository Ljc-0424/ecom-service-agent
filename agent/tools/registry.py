"""集中注册 Agent 可用工具，并提供工具列表和名称映射。"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import BaseTool

from agent.tools.handoff_tools import handoff_to_human
from agent.tools.inventory_tools import get_inventory
from agent.tools.knowledge_tools import search_knowledge_base
from agent.tools.logistics_tools import get_logistics
from agent.tools.order_tools import get_order, get_user_orders
from agent.tools.after_sale_tools import get_after_sale

# 新增 Tool 时在此注册。
_ALL_TOOL_FUNCTIONS: list[Callable[..., Any]] = [
    get_user_orders,
    get_order,
    get_logistics,
    get_inventory,
    get_after_sale,
    search_knowledge_base,
    handoff_to_human,
]


def get_all_tools() -> list[BaseTool]:
    """获取已注册且符合工具接口的业务 Tool。"""
    tools: list[BaseTool] = []
    for fn in _ALL_TOOL_FUNCTIONS:
        if hasattr(fn, "name"):
            tools.append(fn)  # type: ignore[arg-type]
    return tools


def get_tool_map() -> dict[str, BaseTool]:
    """获取工具名到工具对象的映射，供工具节点分发调用。"""
    return {t.name: t for t in get_all_tools()}
