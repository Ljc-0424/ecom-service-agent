"""Agent State：Agent 的「工作台」，保存一次对话要用的结构化信息。

【这个文件在干什么】
  定义 LangGraph 图的状态结构（官方 Quickstart §2 Define state 同款做法：
  TypedDict + Annotated 合并规则），字段为 V1 需求文档定稿的三核心字段
  （messages / orders_context / active_order_id）+ 最小 handoff / trace。

【官方对照】
  官方 Quickstart：
      class MessagesState(TypedDict):
          messages: Annotated[list[AnyMessage], operator.add]
          llm_calls: int
  本项目差异：
      1. messages 用 langgraph 的 add_messages（官方 graph API 惯例，
         按「追加」合并消息，且能按消息 id 去重）
      2. trace 用官方同款 operator.add（官方用 operator.add 给 llm_calls
         计数，我们用同样的手法给 trace 追加记录）
      3. orders_context / active_order_id / handoff 是 V1 需求文档要求的
         业务字段，官方示例没有

【语法速查】
  TypedDict      —— 用「字典」的写法，但有类型提示（IDE 能提示字段）
  Annotated      —— 给字段附加「合并规则」：LangGraph 更新该字段时
                    不整体替换，而是调用规则函数把新旧值合并
  Optional[...]  —— 可以是某种类型，也可以是 None（空）
  total=False    —— TypedDict 的字段都不强制填
"""

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
    """Agent 运行状态。

    【字段与更新规则】
      messages        —— 聊天记录，add_messages 追加合并
      orders_context  —— 对话里出现过的订单摘要（dict，Node 层整体替换新 dict）
      active_order_id —— 当前正在讨论哪笔订单；只由 order_selection
                         事件（apply_order_selection）更新，Tool 不许碰
      handoff         —— 转人工信息；没转人工时为 None
      trace           —— 执行轨迹，operator.add 追加合并（官方同款）
    """

    messages: Annotated[list, add_messages]
    orders_context: dict[str, OrderContext]
    active_order_id: Optional[str]
    handoff: Optional[dict[str, Any]]
    trace: Annotated[list, operator.add]


def empty_state() -> AgentState:
    """造一份「空工作台」，每个字段都给上初始值。

    【为什么单独写这个函数】
      刚开始对话时 State 是空的；写成函数方便复用，
      不用在很多地方手写一长串字典（api/chat.py 和测试都在用）。
    """
    return {
        "messages": [],
        "orders_context": {},
        "active_order_id": None,
        "handoff": None,
        "trace": [],
    }
