"""LangGraph 节点：模型调用、工具执行、分支路由与人工转接。

节点通过工厂函数接收请求级 LLM 和 RuntimeContext，便于隔离会话依赖及测试替身。
工具执行后由节点注入 user_id、合并订单摘要并记录 Trace；active_order_id 仅由
经过校验的订单选择事件更新。
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.runtime.context import RuntimeContext
from agent.state.state import AgentState, OrderContext
from db.session.store import HandoffContext


# ========== 系统提示词 ==========
# V1 裁定：保持基本形态（角色 + 工具说明 + 简短作答原则，≤ 数行）。
# 静态字符串：不拼接 user_id / 订单上下文，行为约束靠机制
# （user_id 注入、order_selection 事件、只读 Tool）而不是 prompt 长清单。
# 仅保留一行路由提示：真实模型评测发现轻量模型遇到「我的手机坏了」这类
# 模糊提问时会反问订单号，而不是先查用户的订单列表。
SYSTEM_PROMPT = """你是电商客服助手。可调用工具查询订单、库存、物流、售后和企业知识库。
实时业务状态以工具查询结果为准，不要凭历史聊天断言。
用户提到自己的订单或商品但未说明订单号时，先查询其订单列表。
无法可靠处理或用户要求人工时，调用 handoff_to_human 转人工。
使用简洁中文回答。"""


# ========== LLM 节点 ==========

def make_llm_node(llm, runtime: RuntimeContext):
    """创建绑定当前请求依赖的 LLM 节点。"""

    def llm_call(state: AgentState) -> dict[str, Any]:
        """调用模型并记录本轮发起的工具调用。"""
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])

        tool_calls = [
            call.get("name")
            for call in getattr(response, "tool_calls", []) or []
            if call.get("name")
        ]
        return {
            "messages": [response],  # 由 add_messages 规则追加进 State
            "trace": [{"type": "llm", "tool_calls": tool_calls}],
        }

    return llm_call


# ========== 条件路由 ==========

def should_continue(state: AgentState) -> Literal["tools", "human_handoff", "end"]:
    """根据模型最后一条消息路由到工具、转人工或结束，不做业务判断。"""
    messages = state["messages"]
    last = messages[-1] if messages else None

    if last is None or not getattr(last, "tool_calls", None):
        return "end"  # 没有工具调用 → 直接用模型的文字回答结束

    if any(c.get("name") == "handoff_to_human" for c in last.tool_calls):
        return "human_handoff"
    return "tools"


# ========== 工具执行 ==========

def make_tool_node(tool_map: dict[str, Any], runtime: RuntimeContext):
    """创建使用当前工具表与请求上下文的工具执行节点。"""

    def tool_node(state: AgentState) -> dict[str, Any]:
        """执行模型请求的工具，生成 ToolMessage 并更新必要的订单上下文。"""
        result: list[ToolMessage] = []
        trace: list[dict[str, Any]] = []
        # Node 层合并：复制旧上下文，把本次工具结果里的订单写进去
        orders_context = dict(state.get("orders_context") or {})

        for tool_call in state["messages"][-1].tool_calls:
            name = tool_call["name"]
            args = dict(tool_call.get("args") or {})
            tool = tool_map.get(name)

            # 身份由请求上下文注入；Service 仍需执行资源归属校验。
            if name in ("get_user_orders", "get_order", "get_logistics", "get_after_sale"):
                args["user_id"] = runtime.user_id

            if tool is None:
                observation: Any = {"success": False, "message": f"未知工具 {name}"}
            else:
                try:
                    observation = tool.invoke(args)
                except Exception as exc:  # 工具报错不让整个 Agent 崩掉
                    observation = {"success": False, "message": str(exc)}

            # ToolMessage.content 使用字符串；tool_call_id 用于关联对应请求。
            result.append(
                ToolMessage(
                    content=observation if isinstance(observation, str) else json.dumps(observation, ensure_ascii=False),
                    tool_call_id=tool_call["id"],
                )
            )
            trace.append(_build_tool_trace(name, args, observation))

            # 查询结果可以补充摘要，但不能隐式切换当前焦点订单。
            if isinstance(observation, dict) and observation.get("success"):
                orders_context = _merge_order_context(orders_context, observation)

        return {"messages": result, "orders_context": orders_context, "trace": trace}

    return tool_node


def _build_tool_trace(
    name: str,
    args: dict[str, Any],
    observation: Any,
) -> dict[str, Any]:
    """生成不暴露完整业务数据的工具轨迹。"""
    event: dict[str, Any] = {
        "type": "tool",
        "name": name,
        "args": args,
    }
    if isinstance(observation, dict):
        success = bool(observation.get("success", True))
        event["success"] = success
        event["result_keys"] = sorted(
            key for key in observation.keys() if key not in {"message"}
        )
        if not success and observation.get("message"):
            event["error"] = str(observation["message"])[:200]
    else:
        event["success"] = True
    return event


def _merge_order_context(orders_context: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Node 层：从工具结果里挑出订单摘要，合并进 orders_context。

    【规则】
      结果里有 order（单笔）→ 记一条；有 orders（列表）→ 每笔记一条；
      同 order_id 覆盖为最新摘要，其它订单保留（V1：多订单不互相覆盖）。
    """
    def _absorb(info: dict[str, Any]) -> None:
        oid = info.get("order_id")
        if not oid:
            return
        orders_context[oid] = OrderContext(
            order_id=oid,
            product_name=info.get("product_name", ""),
            specification=info.get("specification", ""),
            order_status=info.get("order_status", ""),
            summary=f"{info.get('product_name', '')} / {info.get('order_status', '')}",
        )

    if "order" in result and isinstance(result["order"], dict):
        _absorb(result["order"])
    for item in result.get("orders") or []:
        if isinstance(item, dict):
            _absorb(item)
    return orders_context


# ========== 转人工 ==========

def make_handoff_node(runtime: RuntimeContext):
    """创建转人工节点。"""

    def handoff_node(state: AgentState) -> dict[str, Any]:
        """结束自动处理并返回包含必要上下文的转接结果。"""
        reason = ""
        last = state["messages"][-1] if state["messages"] else None
        for call in getattr(last, "tool_calls", []) or []:
            if call.get("name") == "handoff_to_human":
                reason = (call.get("args") or {}).get("reason", "")

        ctx = HandoffContext(
            user_id=runtime.user_id,
            current_question=_last_human_text(state),
            active_order_id=state.get("active_order_id"),
            handoff_reason=reason,
        )
        return {
            "messages": [AIMessage(
                content=f"该问题需要人工客服处理，已为您转接（原因：{reason or '无法自动可靠处理'}）。"
            )],
            "handoff": {
                "user_id": ctx.user_id,
                "reason": ctx.handoff_reason,
                "status": "handed_off",
                "active_order_id": ctx.active_order_id,
                "current_question": ctx.current_question,
            },
            "trace": [{"type": "handoff", "reason": reason}],
        }

    return handoff_node


def _last_human_text(state: AgentState) -> str:
    """从后往前找第一条「用户说的话」（转人工交接单要用）。"""
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""
