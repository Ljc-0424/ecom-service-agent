"""业务 Tool 注册表：所有 Agent 可用工具的「总名单」。

【这个文件在干什么】
  1. _ALL_TOOL_FUNCTIONS —— 名单本身：新增工具时，import 后登记到这里
  2. get_all_tools()     —— 名单 → 工具对象列表（绑定给 LLM 用）
  3. get_tool_map()      —— 名字 → 工具对象（工具节点按名字执行）

【官方对照】（LangGraph Quickstart §1）
  官方写法：
      tools = [add, multiply, divide]
      tools_by_name = {tool.name: tool for tool in tools}
      model_with_tools = model.bind_tools(tools)
  本项目 get_tool_map() 就是官方 tools_by_name 的函数版；
  build_agent_graph 里 bind_tools(list(tool_map.values())) 对应官方 bind_tools(tools)。
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import BaseTool

from agent.tools.handoff_tools import handoff_to_human
from agent.tools.inventory_tools import get_inventory
from agent.tools.knowledge_tools import search_knowledge_base
from agent.tools.logistics_tools import get_logistics
from agent.tools.order_tools import get_order, get_user_orders
from agent.tools.after_sale_tools import get_after_sale

# 名单：加新 Tool 的唯一要改的地方
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
    """获取全部已注册业务 Tool 列表（对应官方 tools = [...]）。

    @tool 装饰器把普通函数变成 BaseTool 对象（带 .name 属性）；
    hasattr 检查是防御：万一有人误把普通函数放进名单，这里过滤掉。
    """
    tools: list[BaseTool] = []
    for fn in _ALL_TOOL_FUNCTIONS:
        if hasattr(fn, "name"):
            tools.append(fn)  # type: ignore[arg-type]
    return tools


def get_tool_map() -> dict[str, BaseTool]:
    """获取工具名 → 工具对象的映射（对应官方 tools_by_name）。

    字典推导式：{t.name: t for t in get_all_tools()}
    工具节点拿模型说的工具名来这里查：tool_map.get(name)。
    """
    return {t.name: t for t in get_all_tools()}
