"""LLM 客户端工厂与确定性测试替身；真实运行配置从 Settings 读取。"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from config.settings import settings


def create_llm(
    temperature: float = 0.0,
    timeout: Optional[float] = None,
    max_retries: int = 2,
) -> BaseChatModel:
    """按应用配置创建真实 LLM 客户端；改写链路可单独设置超时和重试。"""
    if not (settings.llm_api_key or "").strip():
        raise RuntimeError(
            "未配置 LLM_API_KEY，无法进行真实 LLM Tool Calling。"
            "请在 .env 中配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL。"
            "（测试请使用 ScriptedStubLLM，不要用关键词模拟决策。）"
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )


class ScriptedStubLLM(BaseChatModel):
    """测试专用的脚本化替身，不用于评估模型决策质量。

    【剧本长什么样】
      script = [
          {"tool_calls": [{"name": "get_user_orders", "args": {}}]},  # 第1步：要调工具
          {"content": "您好，已查到订单"},                             # 第2步：最终回答
      ]
      每次被调用，按顺序吐出下一条；播完返回一句占位。

    【用来测什么】
      Graph 连线、Tool 执行、ToolMessage 回传、State 更新这些「管线」；
      不测模型的真实决策质量。
    """

    model_name: str = "scripted-stub"
    script: list[dict[str, Any]] = []
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        """模型类型标识，LangChain 内部用。"""
        return "scripted-stub"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ScriptedStubLLM":
        """模拟「绑定工具」：只记录名字，保持接口和真实模型一致。"""
        return ScriptedStubLLM(script=list(self.script), cursor=self.cursor)

    def _generate(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> ChatResult:
        """LangChain 要求实现的「生成一条回复」：按剧本吐下一条。"""
        if self.cursor < len(self.script):
            step = self.script[self.cursor]
            self.cursor += 1
        else:
            step = {"content": "（stub 脚本已执行完毕）"}

        if "tool_calls" in step:
            # 构造「AI 要求调用工具」的消息
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": c["name"],
                        "args": c.get("args") or {},
                        "id": c.get("id") or f"call_stub_{self.cursor}",
                        "type": "tool_call",
                    }
                    for c in step["tool_calls"]
                ],
            )
        else:
            # 普通文字回答
            msg = AIMessage(content=str(step.get("content", "")))

        return ChatResult(generations=[ChatGeneration(message=msg)])
