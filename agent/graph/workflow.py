"""LangGraph 工作流：把节点连成官方 Quickstart 同款的「Agent Loop」。

【这个文件在干什么】
  1. build_agent_graph      —— 官方 §6 同款：建图 + 编译成可执行图
  2. run_agent              —— 跑一轮：用户说一句话 → 得到最终回答
  3. apply_order_selection  —— V1 订单选择结构化事件（前端联动）
  4. extract_final_answer / extract_handoff —— 给 API 层取结果用

【图长什么样（V1 需求文档定稿结构）】
    START
      ↓
    [llm] ──should_continue──→ END（无 tool_calls）
      ↑         │
      │         ├─→ [tools] ──────────┘（有普通 tool_calls，循环）
      │         └─→ [human_handoff] ──→ END（调用了 handoff_to_human）

【官方对照】（LangGraph Quickstart · Graph API §6）
  builder = StateGraph(MessagesState)
  builder.add_node("llm_call", llm_call)
  builder.add_node("tool_node", tool_node)
  builder.add_edge(START, "llm_call")
  builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
  builder.add_edge("tool_node", "llm_call")
  agent = builder.compile()
  本项目差异：节点来自工厂函数（要注入 llm/runtime）、多了 human_handoff 节点、
  State 用带业务字段的 AgentState。
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
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


def build_agent_graph(llm: Any, runtime: RuntimeContext):
    """建图并编译（官方 §6 同款装配）。

    【流程】
      1. get_tool_map() 拿全部业务工具（官方 tools_by_name 惯例）
      2. llm.bind_tools(tools) —— 绑定后模型才知道有哪些工具可调（官方同款）
      3. 加节点、连边、条件路由、编译
    """
    tool_map = get_tool_map()
    llm_with_tools = llm.bind_tools(list(tool_map.values()))

    builder = StateGraph(AgentState)

    # add_node：登记节点；make_xxx 返回真正的节点函数（闭包已捕获依赖）
    builder.add_node("llm", make_llm_node(llm_with_tools, runtime))
    builder.add_node("tools", make_tool_node(tool_map, runtime))
    builder.add_node("human_handoff", make_handoff_node(runtime))

    builder.add_edge(START, "llm")  # 入口先到 LLM

    # 条件边：should_continue 返回的字符串对应下面字典的键（官方同款写法）
    builder.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "human_handoff": "human_handoff",
            "end": END,
        },
    )

    builder.add_edge("tools", "llm")        # 工具做完回 LLM（官方同款循环）
    builder.add_edge("human_handoff", END)  # 转人工则结束（V1 扩展）
    return builder.compile()


def run_agent(
    user_message: str,
    user_id: str,
    state: Optional[AgentState] = None,
    llm: Any = None,
    session_id: str = "",
) -> dict[str, Any]:
    """跑一轮 Agent：用户一句话 → 最终 State（含回答/上下文/轨迹）。

    【签名为什么这样】api/chat.py（前端联动）按这个签名调用，保持不变；
    llm 参数用于测试注入 ScriptedStubLLM，不传则 create_llm() 真模型。

    【官方对照】官方调用方式 agent.invoke({"messages": [HumanMessage(...)]})；
    本项目 State 带业务字段，所以传入完整 State 字典，messages 里追加本轮用户消息。
    """
    if not user_id:
        raise ValueError("user_id 不能为空")  # V1：user_id 必填，不写默认值
    runtime = RuntimeContext(user_id=user_id, session_id=session_id)
    llm = llm if llm is not None else create_llm()
    graph = build_agent_graph(llm=llm, runtime=runtime)

    if state is None:
        state = empty_state()
    else:
        # 复制一份，避免直接改调用方的字典；缺字段补默认
        state = dict(state)
        state.setdefault("messages", [])
        state.setdefault("orders_context", {})
        state.setdefault("active_order_id", None)
        state.setdefault("handoff", None)
        state.setdefault("trace", [])

    # 本轮用户消息追加进 messages（官方同款：传 HumanMessage 进 graph）
    state["messages"] = list(state.get("messages") or []) + [
        HumanMessage(content=user_message)
    ]

    # invoke：执行整张图直到 END；recursion_limit 防模型反复调工具死循环
    return graph.invoke(state, config={"recursion_limit": 25})


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
    倒序遇到的第一条 AIMessage 才是最终回答（官方 agent.invoke 返回同结构）。
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
