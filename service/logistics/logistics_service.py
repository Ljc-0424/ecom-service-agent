"""物流业务 Service：查物流 + 判断是否需要人工关注。

【这个文件在干什么】
  Tool 的「外壳」之下真正干活的层：
  - get_by_order / get_by_tracking —— 查物流
  - needs_human_attention          —— 确定性规则：什么情况要提醒人工

【调用链】
  agent/tools/logistics_tools.py（给 LLM 用）
  api/orders.py（给前端订单面板用）
        ↓ 都调本文件
  db/business/database.py → SQLite

【为什么「实时查物流」很重要】
  历史聊天里说「已发货」可能过时了；物流状态必须现查现答，
  这是 Tool / Service 存在的意义——给模型提供实时事实。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from db.business.database import BusinessDatabase, get_business_db
from db.business.models import LogisticsStatus


@dataclass
class LogisticsInfo:
    """物流业务结果对象：把数据库行转成对外稳定的字段。

    【为什么不用 db 层的 Logistics 类直接返回】
      多一层隔离：以后数据库改字段名，对外结构不变；
      也方便以后加「计算出来的」业务字段（如预计延迟天数）。
      各 Service 的 XxxInfo 都遵循这个模式。
    """

    tracking_number: str  # 快递运单号（如 SF123456）
    order_id: str  # 关联的订单号
    logistics_status: str  # 物流状态：未发货 / 运输中 / 派送中 / 已签收 / 异常 / 退回
    current_location: str  # 当前位置（如「北京分拣中心」）
    estimated_time: str  # 预计送达时间
    shipped_at: Optional[str]  # 实际发货时间（未发货为 None）
    arrived_at: Optional[str]  # 到达目的地时间（未到为 None）
    signed_at: Optional[str]  # 签收时间（未签收为 None）
    sign_info: str  # 签收信息（如「本人签收」）
    is_exception: bool  # 是否运输异常
    exception_reason: str  # 异常原因（正常为空）

    def to_dict(self) -> dict[str, Any]:
        """转成字典，方便塞进 ToolMessage / JSON 响应。"""
        return asdict(self)


class LogisticsService:
    """物流业务逻辑。"""

    def __init__(self, db: Optional[BusinessDatabase] = None) -> None:
        """初始化实例。

        【依赖注入】
          db 参数不传就用全局单例 get_business_db()；
          测试时传独立库，避免污染演示数据。各 Service 同款写法。
        """
        self.db = db or get_business_db()

    def get_by_order(self, order_id: str) -> Optional[LogisticsInfo]:
        """按订单号查询物流。"""
        logi = self.db.get_logistics_by_order(order_id)
        return self._to_info(logi) if logi else None

    def get_by_tracking(self, tracking_number: str) -> Optional[LogisticsInfo]:
        """按运单号查询物流。"""
        logi = self.db.get_logistics_by_tracking(tracking_number)
        return self._to_info(logi) if logi else None

    def get_for_order_context(
        self,
        order_id: Optional[str] = None,
        tracking_number: Optional[str] = None,
    ) -> Optional[LogisticsInfo]:
        """按订单号或运单号查询物流（Tool 层的统一入口）。

        【参数优先级】
          运单号最精确 → 优先用；没有运单号再用订单号；都没有 → None。
          LLM 填参数不一定填哪个，这里统一兜住。
        """
        if tracking_number:
            return self.get_by_tracking(tracking_number)
        if order_id:
            return self.get_by_order(order_id)
        return None

    @staticmethod
    def _to_info(logi) -> LogisticsInfo:
        """将数据库模型转换为业务结果对象。

        【@staticmethod 语法】
          静态方法：不需要访问实例（没有 self 参数），
          只是逻辑上「属于这个类」所以放在类里。
          调用：LogisticsService._to_info(x) 或实例调用都可以。
        """
        return LogisticsInfo(
            tracking_number=logi.tracking_number,
            order_id=logi.order_id,
            logistics_status=logi.logistics_status,
            current_location=logi.current_location,
            estimated_time=logi.estimated_time,
            shipped_at=logi.shipped_at,
            arrived_at=logi.arrived_at,
            signed_at=logi.signed_at,
            sign_info=logi.sign_info,
            is_exception=logi.is_exception,
            exception_reason=logi.exception_reason,
        )

    @staticmethod
    def needs_human_attention(info: LogisticsInfo) -> bool:
        """确定性规则：运输异常 / 退回需要人工介入关注。

        【为什么放 Service 而不是让 LLM 判断】
          「什么算异常」是明确业务规则，代码判断 100% 稳定；
          交给模型猜会时对时错。确定性规则一律放代码里。
        """
        return info.is_exception or info.logistics_status in (
            LogisticsStatus.EXCEPTION,
            LogisticsStatus.RETURNED,
        )
