"""Runtime Context：本次请求的「外部身份信息」，不属于对话内容。

【这个文件在干什么】
  告诉 Agent：「现在是谁在问」。
  例如 user_id = "U001"。

【和 State 的区别】
  State        —— 对话里发生过什么（消息、订单上下文）
  RuntimeContext —— 这次请求本身的固定信息（用户身份）
  user_id 不该让 LLM 猜，也不该写进聊天记录让模型「推理」出来。

【语法速查】
  @dataclass —— 自动生成 __init__ 等样板代码的「数据类」
  frozen=True —— 创建后字段不可改（更安全）
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeContext:
    """一次 Agent 运行时的外部上下文。

    字段：
      user_id    —— 当前用户 ID（必填语义）
      session_id —— 会话 ID（可选，便于多轮对话区分）
    """

    user_id: str
    session_id: str = ""  # 默认空字符串

    def tool_kwargs(self) -> dict[str, str]:
        """给工具调用注入身份参数。

        【流程】
          返回 {"user_id": "..."}，
          调用 get_order / get_user_orders 时合并进参数，
          保证「查的是当前用户的订单」。

        【为什么不用】
          V1 目前在 Tool Node 里直接写 args["user_id"] = runtime.user_id，
          这个方法保留作备选，方便以后统一注入。
        """
        return {"user_id": self.user_id}
