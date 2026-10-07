"""Graph 节点：官方 Quickstart「手搭 Agent」同款三节点 + V1 最小扩展。

【这个文件在干什么】
  LangGraph 图里的节点函数，每个节点：输入当前 State，返回「要更新的字段」：

      START → [llm] → should_continue（条件边）
                        ├─ 无 tool_calls        → END
                        ├─ 有普通 tool_calls     → [tools] → 回 [llm]
                        └─ handoff_to_human 调用 → [human_handoff] → END

【官方对照】（LangGraph Quickstart · Graph API）
  §3 Define model node   → llm_call
  §4 Define tool node    → tool_node
  §5 Define end logic    → should_continue
  官方原文（tool_node）：
      for tool_call in state["messages"][-1].tool_calls:
          tool = tools_by_name[tool_call["name"]]
          observation = tool.invoke(tool_call["args"])
          result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

【V1 差异（官方没有、需求文档要求的，共 4 处）】
  1. should_continue 三路：多了 human_handoff（V1 需求的转人工）
  2. tool_node 执行前注入 user_id（V1 安全约束：身份系统注入，不让 LLM 猜）
  3. tool_node 所在 Node 层把成功结果里的订单摘要合并进 orders_context
     （V1 裁定：Tool 不改 State；Node 层决定写入；active_order_id 不在此更新）
  4. 每个节点追加一条最小 trace（V1 可观察性要求）

【为什么用工厂函数 make_xxx】
  官方示例的节点直接引用模块级全局 llm_with_tools；
  本项目要把 llm / runtime 作为参数传入（测试要注入 Stub、user_id 要按请求变），
  所以用「外层函数准备依赖 → 返回闭包节点」的工厂写法，节点函数体逐行同官方。
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
# HumanMessage 用户说的；AIMessage 模型说的（可能带 tool_calls）
# ToolMessage 工具返回的结果；SystemMessage 系统提示词

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


# ========== LLM 节点（官方 §3 同款） ==========

def make_llm_node(llm, runtime: RuntimeContext):
    """工厂：造出 LLM 节点函数（官方节点引用全局变量，这里参数化传入）。"""

    def llm_call(state: AgentState) -> dict[str, Any]:
        """LLM 节点：决定「直接回答」还是「调用工具」。

        【官方对照】Quickstart §3，逐行同款：
            model_with_tools.invoke([SystemMessage(...)] + state["messages"])
        【V1 差异】SYSTEM_PROMPT 是静态基本提示词；追加一条最小 trace。
        """
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])

        return {
            "messages": [response],  # 由 add_messages 规则追加进 State
            "trace": [{"type": "llm"}],
        }

    return llm_call


# ========== 条件边（官方 §5 同款 + 转人工分支） ==========

def should_continue(state: AgentState) -> Literal["tools", "human_handoff", "end"]:
    """看模型最后一条消息的 tool_calls 决定去哪（纯路由，不做业务判断）。

    【官方对照】Quickstart §5：官方两路 Literal["tool_node", END]；
    【V1 差异】多了 human_handoff 一路：模型调用了 handoff_to_human
    就停止自动对话（V1 需求：一旦决定人工接管，不继续自动流程）。
    """
    messages = state["messages"]
    last = messages[-1] if messages else None

    if last is None or not getattr(last, "tool_calls", None):
        return "end"  # 没有工具调用 → 直接用模型的文字回答结束

    # any()：只要有一个 handoff_to_human 调用，就走转人工
    if any(c.get("name") == "handoff_to_human" for c in last.tool_calls):
        return "human_handoff"
    return "tools"


# ========== 工具节点（官方 §4 同款 + 3 处 V1 扩展） ==========

def make_tool_node(tool_map: dict[str, Any], runtime: RuntimeContext):
    """工厂：造出工具节点函数（官方 tool_map 同款结构，参数化传入）。"""

    def tool_node(state: AgentState) -> dict[str, Any]:
        """执行模型点名的工具，每个 tool_call 生成一条 ToolMessage。

        【官方对照】Quickstart §4，循环结构逐行同款。
        【V1 差异】
          1. get_user_orders / get_order 执行前注入 user_id（安全边界）
          2. Node 层把成功结果的订单摘要合并进 orders_context
          3. 追加最小 trace
        """
        result: list[ToolMessage] = []
        trace: list[dict[str, Any]] = []
        # Node 层合并：复制旧上下文，把本次工具结果里的订单写进去
        orders_context = dict(state.get("orders_context") or {})

        for tool_call in state["messages"][-1].tool_calls:
            name = tool_call["name"]
            args = dict(tool_call.get("args") or {})
            tool = tool_map.get(name)

            # V1 扩展 1：user_id 系统注入，不依赖模型自觉
            # 覆盖全部携带订单/身份语义的工具，Service 层再做归属校验兜底
            if name in ("get_user_orders", "get_order", "get_logistics", "get_after_sale"):
                args["user_id"] = runtime.user_id

            if tool is None:
                observation: Any = {"success": False, "message": f"未知工具 {name}"}
            else:
                try:
                    observation = tool.invoke(args)  # 官方同款：tool.invoke(args)
                except Exception as exc:  # 工具报错不让整个 Agent 崩掉
                    observation = {"success": False, "message": str(exc)}

            # ToolMessage.content 需要 str；dict 结果转 JSON（官方 ToolMessage 同款关联方式）
            result.append(
                ToolMessage(
                    content=observation if isinstance(observation, str) else json.dumps(observation, ensure_ascii=False),
                    tool_call_id=tool_call["id"],
                )
            )
            trace.append({"type": "tool", "name": name, "args": args})

            # V1 扩展 2：Node 层合并订单摘要（active_order_id 不在这里动）
            if isinstance(observation, dict) and observation.get("success"):
                orders_context = _merge_order_context(orders_context, observation)

        return {"messages": result, "orders_context": orders_context, "trace": trace}

    return tool_node


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


# ========== 转人工节点（V1 最小自实现，官方无此示例） ==========

def make_handoff_node(runtime: RuntimeContext):
    """工厂：造出转人工节点函数。"""

    def handoff_node(state: AgentState) -> dict[str, Any]:
        """停止自动对话，留下交接信息，给用户一句转接答复。

        【为什么存在】V1 需求：Agent 判断无法可靠处理时转人工；
        转人工不是简单说「请联系客服」，要带上交接信息。
        【流程】从最后一条 AI 消息的 handoff_to_human 调用里取 reason
        → 组装最小 handoff 信息（V1 裁定 4 字段）→ 返回转接答复。
        """
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
