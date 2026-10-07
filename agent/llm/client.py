"""LLM 封装：真实模型入口 + 测试用假模型。

【这个文件在干什么】
  1. create_llm()     —— 正式运行：连真实大模型（必须配 API Key）
  2. ScriptedStubLLM  —— 只给测试用：按预写好的「剧本」返回消息

【官方对照】
  官方 OpenAI 集成页写法：
      from langchain_openai import ChatOpenAI
      model = ChatOpenAI(model="...", api_key=..., base_url=..., temperature=0)
  本项目差异：
      1. model / api_key / base_url 从 config.settings 读（V1 需求 §27：
         配置走环境变量，禁止硬编码密钥）
      2. Key 为空直接抛 RuntimeError —— V1 硬约束：运行时必须真实 LLM，
         禁止无 Key 时静默回退假模型
      3. ScriptedStubLLM 是官方没有的测试替身（V1 裁定：测试可用
         「脚本化 tool_calls」的 Stub，但禁止关键词假决策），最小实现

【V1 重要约定】
  工具选择必须由真实 LLM 的 Tool Calling 决定；
  Stub 只是「照剧本念台词」，不代表模型能力，且只在 tests 里用。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from config.settings import settings


def create_llm(temperature: float = 0.0) -> BaseChatModel:
    """创建真实 LLM 客户端（官方 OpenAI 集成同款写法）。

    【流程】
      1. 读配置里的 API Key
      2. Key 为空 → 直接报错（不许用假模型冒充真实运行）
      3. 有 Key → 创建 ChatOpenAI，后续 bind_tools 绑定工具

    【参数】
      temperature —— 0 表示回答更稳定（官方示例同款取值）
    """
    if not (settings.llm_api_key or "").strip():
        raise RuntimeError(
            "未配置 LLM_API_KEY，无法进行真实 LLM Tool Calling。"
            "请在 .env 中配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL。"
            "（测试请使用 ScriptedStubLLM，不要用关键词模拟决策。）"
        )

    from langchain_openai import ChatOpenAI  # 延迟导入：用到真实模型才加载

    return ChatOpenAI(
        model=settings.llm_model,        # 模型名（官方参数）
        api_key=settings.llm_api_key,    # 密钥（官方参数）
        base_url=settings.llm_base_url,  # 接口地址，支持 OpenAI 兼容端点（官方参数）
        temperature=temperature,
    )


class ScriptedStubLLM(BaseChatModel):
    """测试专用：按「剧本」返回消息（官方没有，V1 允许的最小测试替身）。

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
    # default_factory=list：每个实例各建一个新列表（不能写 =[]，会共享）
    script: list[dict[str, Any]] = []
    cursor: int = 0  # 播到剧本第几条了

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
                        "name": c["name"],            # 工具名
                        "args": c.get("args") or {},  # 参数
                        "id": c.get("id") or f"call_stub_{self.cursor}",  # 调用编号
                        "type": "tool_call",
                    }
                    for c in step["tool_calls"]
                ],
            )
        else:
            # 普通文字回答
            msg = AIMessage(content=str(step.get("content", "")))

        return ChatResult(generations=[ChatGeneration(message=msg)])
