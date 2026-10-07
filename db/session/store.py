"""会话存储：V1 仅保留当前会话内存状态。

【这个文件在干什么】
  两个东西：
  1. SessionStore        —— 进程内的「会话档案柜」：谁在哪次对话里聊了什么摘要
  2. HandoffContext      —— 转人工时的「交接单」：人工客服接手前需要知道的信息

【谁在用】
  api/chat.py     —— 每轮对话结束后把摘要写进 SessionStore（_persist_session）
  graph/nodes.py  —— 转人工节点组装 HandoffContext（人工侧读取是 POST-V1 的事）

【V1 的简化】
  数据只放在内存字典里：进程重启就没了，也不支持多进程部署。
  [POST-V1] 会话持久化 / Checkpoint Storage
  [POST-V1] 长期 Memory / 用户画像
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class HandoffContext:
    """人工交接单：转人工时打包给人工客服的上下文。

    字段：
      user_id                —— 哪个用户被转接
      current_question       —— 用户最后问了什么（人工接手先看这个）
      active_order_id        —— 当时正在讨论的订单
      relevant_order_context —— 相关订单摘要列表
      handoff_reason         —— 为什么转人工
    """

    user_id: str  # 被转接的用户编号
    current_question: str = ""  # 用户最后问了什么（人工接手先看这个）
    active_order_id: Optional[str] = None  # 当时正在讨论的订单
    relevant_order_context: list[dict[str, Any]] = field(default_factory=list)  # 相关订单摘要
    handoff_reason: str = ""  # 转人工原因

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化字典（asdict 是 dataclass 自带的转字典方法）。"""
        return asdict(self)


@dataclass
class ConversationRecord:
    """一场对话的档案（内存版）：会话 ID + 用户 + 各类摘要。

    注意字段都是「摘要」不是全量：messages 只留最近 50 条、
    每条截断到 2000 字，V1 用途是调试观察，不是完整日志。
    """

    session_id: str  # 会话编号（如 S3f8a2c91d40b）
    user_id: str  # 用户编号
    # default_factory=list/dict：每个实例各建一个新列表/字典，不共享
    messages: list[dict[str, Any]] = field(default_factory=list)  # 最近消息摘要（只留 50 条）
    orders_context: dict[str, Any] = field(default_factory=dict)  # 会话里出现过的订单
    active_order_id: Optional[str] = None  # 当前讨论的订单
    handoff: Optional[dict[str, Any]] = None  # 转人工信息（没转为 None）
    traces: list[dict[str, Any]] = field(default_factory=list)  # 执行轨迹


class SessionStore:
    """进程内会话存储：用一个字典当「数据库」。

    单请求模型：无锁、无落盘，进程重启数据清空。
    """

    _default: Optional["SessionStore"] = None

    def __init__(self) -> None:
        """初始化内存会话表。"""
        # 键 = session_id，值 = 该会话的 ConversationRecord
        self._sessions: dict[str, ConversationRecord] = {}

    @classmethod
    def get_default(cls) -> "SessionStore":
        """获取全局会话存储单例（第一次调用时创建）。

        【classmethod 单例模式】和 VectorStore.get_default 同款写法：
        类变量 _default 存全局那一份，没建过就建一个。
        """
        if cls._default is None:
            cls._default = SessionStore()
        return cls._default

    @classmethod
    def set_default(cls, store: "SessionStore") -> None:
        """替换全局会话存储（测试用）。"""
        cls._default = store

    def create_session(self, user_id: str, session_id: str) -> ConversationRecord:
        """创建（或覆盖）一条会话记录。"""
        rec = ConversationRecord(session_id=session_id, user_id=user_id)
        self._sessions[session_id] = rec
        return rec

    def get_session(self, session_id: str) -> Optional[ConversationRecord]:
        """按会话 ID 获取记录（没有返回 None）。"""
        return self._sessions.get(session_id)

    def save_session(self, rec: ConversationRecord) -> None:
        """保存/更新会话记录（字典按键覆盖，天然是 upsert）。"""
        self._sessions[rec.session_id] = rec

    def list_sessions(self, user_id: Optional[str] = None) -> list[ConversationRecord]:
        """列出所有会话；传了 user_id 就只列该用户的。

        【列表推导式 + if 过滤】
          [s for s in items if s.user_id == user_id]：
          从 items 里挑出满足条件的元素组成新列表。
        """
        items = list(self._sessions.values())
        if user_id:
            items = [s for s in items if s.user_id == user_id]
        return items
