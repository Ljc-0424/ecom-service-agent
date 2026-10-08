"""LangGraph State 定义：维护消息、订单上下文、当前订单、转人工信息和 Trace。"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class OrderContext(TypedDict, total=False):
    """一笔订单在对话中的摘要（键可缺省）。

    orders_context 的值类型：以 order_id 为键，多订单并存、互不覆盖
    （V1 需求文档 §7 的硬性约束）。
    """

    order_id: str  # 订单号（如 A001）
    product_name: str  # 商品名
    specification: str  # 规格
    order_status: str  # 订单状态（如 待发货）
    summary: str  # 一句话摘要（「商品名 / 状态」，给模型和前端看）


class AgentState(TypedDict, total=False):
    """Agent State；订单焦点仅由校验后的订单选择事件更新。"""

    messages: Annotated[list, add_messages]
    orders_context: dict[str, OrderContext]
    active_order_id: Optional[str]
    handoff: Optional[dict[str, Any]]
    trace: Annotated[list, operator.add]


def empty_state() -> AgentState:
    """创建包含全部默认字段的空 Agent State。"""
    return {
        "messages": [],
        "orders_context": {},
        "active_order_id": None,
        "handoff": None,
        "trace": [],
    }
