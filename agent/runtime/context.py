"""请求级上下文：承载不属于对话内容的用户身份与会话标识。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeContext:
    """一次 Agent 执行使用的身份和会话上下文。"""

    user_id: str
    session_id: str = ""  # 默认空字符串

    def tool_kwargs(self) -> dict[str, str]:
        """返回可注入工具参数的身份字段。"""
        return {"user_id": self.user_id}
