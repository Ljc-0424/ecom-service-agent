"""LangGraph 工作流装配、同步/流式运行及订单选择事件处理。

会话 API 通过 InMemorySaver 和 thread_id 保存多轮状态；测试与评测可无状态运行。
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.graph.nodes import (
    make_handoff_node,
    make_llm_node,
    make_tool_node,
    should_continue,
)
from agent.llm.client import create_llm
from agent.runtime.context import RuntimeContext
from agent.state.state import AgentState, OrderContext, empty_state
from agent.tools.registry import get_tool_map

# 检查点仅保存在进程内，服务重启后会话状态不会保留。
_CHECKPOINTER = InMemorySaver()


def build_agent_graph(llm: Any, runtime: RuntimeContext, checkpointer: Any = None):
    """装配 Agent 状态图；可选启用会话检查点。"""
    tool_map = get_tool_map()
    llm_with_tools = llm.bind_tools(list(tool_map.values()))

    builder = StateGraph(AgentState)

    builder.add_node("llm", make_llm_node(llm_with_tools, runtime))
    builder.add_node("tools", make_tool_node(tool_map, runtime))
    builder.add_node("human_handoff", make_handoff_node(runtime))

    builder.add_edge(START, "llm")

    builder.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "human_handoff": "human_handoff",
            "end": END,
        },
    )

    builder.add_edge("tools", "llm")
    builder.add_edge("human_handoff", END)
    return builder.compile(checkpointer=checkpointer)


def _build_runtime_parts(
    user_id: str,
    session_id: str,
    llm: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """run_agent / stream_agent_events 共用的准备逻辑。

    返回 (graph, input_state, config)：
      session_id 非空（API 路径）→ 启用 Checkpointer，thread_id = session_id，
        input_state 只含「本轮增量」（handoff 清零；新消息由调用方追加）；
      session_id 为空（测试/评测路径）→ 无状态运行，input_state 是完整初始 State。
    """
    if not user_id:
        raise ValueError("user_id 不能为空")  # V1：user_id 必填，不写默认值
    runtime = RuntimeContext(user_id=user_id, session_id=session_id)
    llm = llm if llm is not None else create_llm()

    if session_id:
        graph = build_agent_graph(llm=llm, runtime=runtime, checkpointer=_CHECKPOINTER)
        config: dict[str, Any] = {
            "recursion_limit": 25,
            "configurable": {"thread_id": session_id},
        }
        return graph, {"handoff": None}, config  # 每轮清掉上一轮的转人工标记

    graph = build_agent_graph(llm=llm, runtime=runtime)
    return graph, _prepare_state(None), {"recursion_limit": 25}


def run_agent(
    user_message: str,
    user_id: str,
    state: Optional[AgentState] = None,
    llm: Any = None,
    session_id: str = "",
) -> dict[str, Any]:
    """跑一轮 Agent：用户一句话 → 最终 State（含回答/上下文/轨迹）。

    【两种模式】
      session_id 非空（API 路径）→ Checkpointer + thread_id 多轮持久化：
        state 参数是「增量」（如订单选择事件的更新字段），合并进会话状态，
        不传则只追加本轮消息——历史状态由检查点自动延续；
      session_id 为空（测试/评测路径）→ 无状态运行，state 是完整初始 State。

    """
    graph, input_state, config = _build_runtime_parts(user_id, session_id, llm)

    if state:
        # 合并调用方给的增量字段（orders_context / active_order_id / handoff 等）
        for key, value in state.items():
            input_state[key] = value

    input_state["messages"] = list(input_state.get("messages") or []) + [
        HumanMessage(content=user_message)
    ]

    # 限制图的最大执行步数，避免模型反复调用工具。
    return graph.invoke(input_state, config=config)


def _prepare_state(state: Optional[AgentState]) -> AgentState:
    """准备初始 State：空工作台或复制补齐字段（invoke 与流式两条路共用）。"""
    if state is None:
        return empty_state()
    new_state = dict(state)  # 复制一份，避免直接改调用方的字典
    new_state.setdefault("messages", [])
    new_state.setdefault("orders_context", {})
    new_state.setdefault("active_order_id", None)
    new_state.setdefault("handoff", None)
    new_state.setdefault("trace", [])
    return new_state


def stream_agent_events(
    user_message: str,
    user_id: str,
    state: Optional[AgentState] = None,
    llm: Any = None,
    session_id: str = "",
):
    """流式版 run_agent：边跑边产出事件，不直接返回最终 State。

    【产出的事件（dict）】
      {"type": "status", "stage": "tools", "tools": ["get_logistics"]}
        —— 工具节点开始执行时通知（前端显示「正在查询物流…」）
      {"type": "token", "delta": "您"}
        —— 模型最终回答的一个字/词（前端逐字渲染）
      {"type": "final", "result": <完整 State>}
        —— 流结束后的最终 State（调用方据此更新会话缓存与响应字段）

    """
    if not user_id:
        raise ValueError("user_id 不能为空")
    runtime = RuntimeContext(user_id=user_id, session_id=session_id)
    llm = llm if llm is not None else create_llm()
    graph, input_state, config = _build_runtime_parts(user_id, session_id, llm)

    if state:
        for key, value in state.items():
            input_state[key] = value

    # 有 Checkpointer 时，先读出检查点里的已有状态作为累计底座
    # （无状态路径的底座就是完整输入本身）
    final: dict[str, Any] = dict(input_state)
    if config.get("configurable"):
        snapshot = graph.get_state(config)
        base = dict(snapshot.values or {})
        for key in ("orders_context", "active_order_id", "handoff", "messages", "trace"):
            if key in base and key not in input_state:
                final[key] = base[key]
    final["messages"] = list(final.get("messages") or []) + [
        HumanMessage(content=user_message)
    ]
    for mode, chunk in graph.stream(
        input_state, config=config, stream_mode=["updates", "messages"]
    ):
        if mode == "updates":
            for node_name, update in (chunk or {}).items():
                if not isinstance(update, dict):
                    continue
                # 状态字段：最后写入胜出；messages/trace：追加
                for key in ("orders_context", "active_order_id", "handoff"):
                    if key in update:
                        final[key] = update[key]
                if "messages" in update:
                    final["messages"] = list(final.get("messages") or []) + list(update["messages"])
                if "trace" in update:
                    final["trace"] = list(final.get("trace") or []) + list(update["trace"])
                # 工具节点跑完 → 通知前端本轮执行了哪些工具
                if node_name == "tools":
                    names = [t.get("name") for t in (update.get("trace") or []) if t.get("type") == "tool"]
                    if names:
                        yield {"type": "status", "stage": "tools", "tools": names}
        elif mode == "messages":
            msg = chunk[0] if isinstance(chunk, tuple) else chunk
            # 只要模型输出的文字增量；工具消息/带 tool_calls 的空块跳过
            if getattr(msg, "type", "") == "AIMessageChunk":
                delta = getattr(msg, "content", "")
                if isinstance(delta, str) and delta:
                    yield {"type": "token", "delta": delta}
    yield {"type": "final", "result": final}


def apply_order_selection(
    state: AgentState,
    order_id: str,
    runtime: RuntimeContext,
) -> dict[str, Any]:
    """前端「点选订单」事件的 State 更新（V1 结构化事件，前端联动）。

    【完整流程】
      前端点击订单 → POST /api/order-selection {"type":"order_selection", ...}
      → 本函数校验订单属于当前用户
        ├─ 不是 → 返回 {}（拒绝）
        └─ 是   → 更新 orders_context + active_order_id → Agent 继续回答

    【为什么不让 Tool 改 active_order_id】
      V1 裁定：「当前在讨论哪单」只能走校验后的结构化事件更新，
      查库工具不许顺手改 State（防止 LLM 的查询行为改变业务状态）。
    """
    from service.order.order_service import OrderService

    service = OrderService()
    info = service.get_order(order_id=order_id, user_id=runtime.user_id)
    if info is None:
        return {}  # 订单不存在或不属于该用户

    orders_context = dict(state.get("orders_context") or {})
    orders_context[order_id] = OrderContext(
        order_id=order_id,
        product_name=info.product_name,
        specification=info.specification,
        order_status=info.order_status,
        summary=f"{info.product_name} / {info.order_status}",
    )
    return {
        "orders_context": orders_context,
        "active_order_id": order_id,
        "trace": [{"type": "order_selection", "order_id": order_id}],
    }


def extract_final_answer(result: dict[str, Any]) -> str:
    """从运行结果里取「模型最后一次说的正文」（api/chat.py 在用）。

    【为什么从后往前】messages 后段是 ToolMessage / 中间回复；
    倒序遇到的第一条 AIMessage 是最终回答。
    """
    for m in reversed(result.get("messages") or []):
        if isinstance(m, AIMessage):
            content = m.content
            if isinstance(content, list):
                # 有的模型 content 是分块列表，拼成字符串
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            return str(content or "")
    return ""


def extract_handoff(result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """取出转人工信息；没有则 None（api/chat.py 在用）。"""
    return result.get("handoff")
