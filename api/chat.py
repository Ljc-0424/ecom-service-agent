"""Chat API：浏览器和 Agent 之间的桥。

【这个文件在干什么】
  前端只发 HTTP；不参与 Agent 内部怎么想。

  POST /api/chat           —— 用户说话，返回机器人回答
  POST /api/order-selection —— 用户点选了某笔订单
  GET  /api/chat/{id}/state —— 看当前会话状态（调试用）

【一次聊天的流程】
  前端 fetch
    ↓
  chat() 校验 user_id
    ↓
  找到/新建会话 State
    ↓
  run_agent() 跑完整图
    ↓
  把回答、订单上下文、是否要选订单…打包返回 JSON

【语法速查】
  APIRouter()     —— 一组接口的集合，最后挂到主 app 上
  @router.post()  —— 声明「POST 路径」的接口
  class ChatRequest(BaseModel) —— 请求体格式（自动校验）
  HTTPException   —— 抛出后变成 400/404 等 HTTP 错误
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.graph.workflow import (
    apply_order_selection,
    extract_final_answer,
    extract_handoff,
    run_agent,
)
from agent.runtime.context import RuntimeContext
from db.session.store import SessionStore

router = APIRouter()

# 进程里的会话缓存：session_id → AgentState 快照
# V1 单进程内存模型；重启就没了（POST-V1 再考虑持久化）
_SESSION_STATES: dict[str, dict[str, Any]] = {}


class ChatRequest(BaseModel):
    """聊天请求体：前端 POST 时的 JSON 字段。"""

    message: str = Field(..., description="用户消息")
    # ... 表示必填
    user_id: str = Field(..., description="当前用户 ID，由前端显式传入")
    session_id: str = Field(default="", description="会话 ID，空则新建")


class OrderSelectionRequest(BaseModel):
    """订单选择事件请求体。"""

    type: str = Field(default="order_selection")
    order_id: str
    user_id: str = Field(..., description="当前用户 ID")
    session_id: str = ""


class ChatResponse(BaseModel):
    """聊天响应：给前端的 JSON。

    字段的前端用途：
      reply                —— 机器人的最终回答（直接显示）
      need_order_selection —— True 时前端弹出「选择订单」列表：
                              用户名下有多单、又没说清问哪单
      candidate_orders     —— 弹窗里展示的候选订单摘要
      orders_context       —— 对话里出现过的订单（侧栏展示用）
      handoff              —— 非空表示已转人工，前端显示转接提示
      trace                —— 本轮执行轨迹（调试/演示用）
    """

    session_id: str
    reply: str
    active_order_id: Optional[str] = None
    orders_context: dict[str, Any] = Field(default_factory=dict)
    need_order_selection: bool = False
    candidate_orders: list[dict[str, Any]] = Field(default_factory=list)
    handoff: Optional[dict[str, Any]] = None
    trace: list[dict[str, Any]] = Field(default_factory=list)


def _get_or_create_session(user_id: str, session_id: str) -> tuple[str, dict[str, Any]]:
    """拿会话；没有就建一个空 State。

    【流程】
      1. 已有 session_id 且在缓存 → 直接用
      2. 否则生成新 id，注册到 SessionStore，并准备空 State

    【返回】
      (会话ID, 该会话的 State 字典)
    """
    store = SessionStore.get_default()
    if session_id and session_id in _SESSION_STATES:
        return session_id, _SESSION_STATES[session_id]

    if not session_id:
        session_id = f"S{uuid.uuid4().hex[:12]}"  # 随机短 ID
    if store.get_session(session_id) is None:
        store.create_session(user_id=user_id, session_id=session_id)

    state = {
        "messages": [],
        "orders_context": {},
        "active_order_id": None,
        "handoff": None,
        "trace": [],
    }
    _SESSION_STATES[session_id] = state
    return session_id, state


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """对话主接口。

    【流程】
      1. 校验 user_id
      2. 取会话 State；本轮先清掉上一次的 handoff（避免粘滞）
      3. run_agent() 真正跑 Agent
      4. 取最终回答 / 转人工信息 / 订单上下文
      5. 若有多订单且未选定 → 告诉前端展示候选订单
      6. 返回 ChatResponse
    """
    if not req.user_id:
        raise HTTPException(status_code=400, detail="user_id 不能为空")
    sid, state = _get_or_create_session(req.user_id, req.session_id)
    state = dict(state)
    state["handoff"] = None
    prev_trace_len = len(state.get("trace") or [])  # 记下旧轨迹长度，方便只返回本轮

    try:
        result = run_agent(
            user_message=req.message,
            user_id=req.user_id,
            state=state,
            session_id=sid,
        )
    except Exception as exc:  # noqa: BLE001
        # 捕获异常并明确报错；V1 不做自动重试
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {exc}") from exc

    _SESSION_STATES[sid] = result
    _persist_session(sid, req.user_id, result)

    final = extract_final_answer(result)
    handoff = extract_handoff(result)
    orders_context = result.get("orders_context") or {}
    active = result.get("active_order_id")
    all_trace = result.get("trace") or []
    turn_trace = all_trace[prev_trace_len:]

    candidates = []
    need_selection = False
    if not active and len(orders_context) > 1:
        need_selection = True
        candidates = [
            {
                "order_id": ctx.get("order_id"),
                "product_name": ctx.get("product_name"),
                "order_status": ctx.get("order_status"),
            }
            for ctx in orders_context.values()
        ]

    return ChatResponse(
        session_id=sid,
        reply=final,
        active_order_id=active,
        orders_context=orders_context,
        need_order_selection=need_selection,
        candidate_orders=candidates,
        handoff=handoff,
        trace=turn_trace,
    )


@router.post("/order-selection", response_model=ChatResponse)
def order_selection(req: OrderSelectionRequest) -> ChatResponse:
    """用户在前端点了某笔订单。

    【流程】
      1. 校验 type / user_id
      2. apply_order_selection：查这单是否合法，合法则更新 active_order_id
      3. 再跑一轮 Agent，让模型知道「现在在谈哪单」
    """
    if req.type != "order_selection":
        raise HTTPException(status_code=400, detail="type 必须为 order_selection")
    if not req.user_id:
        raise HTTPException(status_code=400, detail="user_id 不能为空")

    sid, state = _get_or_create_session(req.user_id, req.session_id)
    runtime = RuntimeContext(user_id=req.user_id, session_id=sid)
    update = apply_order_selection(state, req.order_id, runtime)
    if not update:
        raise HTTPException(status_code=400, detail=f"无效订单 {req.order_id}")

    new_state = {**state, **update}  # 字典合并：旧 State + 更新字段
    _SESSION_STATES[sid] = new_state

    result = run_agent(
        user_message=f"（系统事件）用户已选择订单 {req.order_id}，请围绕该订单继续。",
        user_id=req.user_id,
        state=new_state,
        session_id=sid,
    )
    _SESSION_STATES[sid] = result
    _persist_session(sid, req.user_id, result)

    return ChatResponse(
        session_id=sid,
        reply=extract_final_answer(result),
        active_order_id=result.get("active_order_id"),
        orders_context=result.get("orders_context") or {},
        need_order_selection=False,
        candidate_orders=[],
        handoff=extract_handoff(result),
        trace=result.get("trace") or [],
    )


@router.get("/chat/{session_id}/state")
def get_chat_state(session_id: str) -> dict:
    """调试用：看某会话当前 active_order_id / orders_context。"""
    state = _SESSION_STATES.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="session not found")
    return {
        "session_id": session_id,
        "active_order_id": state.get("active_order_id"),
        "orders_context": state.get("orders_context") or {},
        "handoff": state.get("handoff"),
    }


def _persist_session(sid: str, user_id: str, result: dict[str, Any]) -> None:
    """把本轮结果摘要放进内存 SessionStore（方便以后扩展）。

    【流程】
      找到/创建会话记录 → 写入订单上下文、handoff、精简消息列表
    """
    store = SessionStore.get_default()
    rec = store.get_session(sid) or store.create_session(user_id=user_id, session_id=sid)
    rec.user_id = user_id
    rec.orders_context = result.get("orders_context") or {}
    rec.active_order_id = result.get("active_order_id")
    rec.handoff = result.get("handoff")
    rec.traces = result.get("trace") or []
    msgs = []
    for m in result.get("messages") or []:
        content = getattr(m, "content", "")
        if not isinstance(content, str):
            content = str(content)
        msgs.append({"type": getattr(m, "type", "unknown"), "content": content[:2000]})
    rec.messages = msgs[-50:]  # 只留最近 50 条摘要
    store.save_session(rec)
