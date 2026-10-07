"""转人工 Tool：特殊能力，不是普通业务查询。

【官方对照】@tool + Args docstring，同 order_tools.py 文件头说明。
【名字拆解】handoff_to_human = hand off（移交）+ to human（给人工）=「转人工」。
【语义】Agent 自己处理不了 / 用户明确要人工 / 涉及禁止操作 → 调这个工具
  → should_continue 路由到 human_handoff 节点 → 停止自动对话。
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def handoff_to_human(reason: str) -> dict:
    """请求转人工客服。以下情况必须调用本工具，不要再自行回复用户：

    1. 用户明确要求人工客服 / 真人客服；
    2. 问题涉及退款执行、资金操作、投诉升级等无法自动可靠处理的操作；
    3. 已尝试查询仍无法给出可靠答复。

    Args:
        reason: 转人工的原因
    """
    return {
        "success": True,
        "handoff": True,
        "reason": reason,
        "status": "handed_off",
        "message": "已为您转接人工客服，请稍候。",
    }
