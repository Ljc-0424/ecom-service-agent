"""售后业务 Service。第一版支持查询售后/退款状态。

【职责边界】
  只查询：V1 不做自动退款、自动改状态——退款这类敏感操作
  必须走人工（这正是 handoff_to_human 工具存在的理由之一）。

【调用链】
  agent/tools/after_sale_tools.py（给 LLM 用）→ 本文件 → SQLite
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from db.business.database import BusinessDatabase, get_business_db


@dataclass
class AfterSaleInfo:
    """售后业务结果对象：数据库行的对外稳定视图（模式同 LogisticsInfo）。"""

    after_sale_id: str  # 售后单号（如 AS001）
    order_id: str  # 关联的订单号
    after_sale_type: str  # 售后类型：退款 / 退货 / 换货 / 维修
    after_sale_status: str  # 处理进度：待处理 / 处理中 / 已完成 / 已拒绝
    refund_status: str  # 退款进度：无 / 退款处理中 / 退款成功 / 退款失败
    refund_amount: float  # 退款金额（0 = 未设定）
    created_at: str  # 售后单创建时间
    updated_at: str  # 最近一次状态变更时间
    reason: str  # 售后原因（用户填写）

    def to_dict(self) -> dict[str, Any]:
        """转成字典，方便塞进 ToolMessage / JSON 响应。"""
        return asdict(self)


class AfterSaleService:
    """售后业务逻辑。"""

    def __init__(self, db: Optional[BusinessDatabase] = None) -> None:
        """初始化：默认用全局业务库，测试时可注入独立库。"""
        self.db = db or get_business_db()

    def get_after_sale(
        self,
        after_sale_id: Optional[str] = None,
        order_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[AfterSaleInfo]:
        """按售后单号或订单号查询售后。

        【查询策略】
          售后单号最精确 → 优先；没有再用订单号（一笔订单通常只有
          一条售后记录）；两个都查不到 → None。

        【归属校验（安全边界）】
          传了 user_id 就校验售后单所属订单的用户：查别人的售后一律 None。
        """
        aft = None
        if after_sale_id:
            aft = self.db.get_after_sale(after_sale_id)
        if aft is None and order_id:
            aft = self.db.get_after_sale_by_order(order_id)
        if aft is None:
            return None
        if user_id:
            order = self.db.get_order(aft.order_id)
            if order is None or order.user_id != user_id:
                return None
        return self._to_info(aft)

    @staticmethod
    def _to_info(aft) -> AfterSaleInfo:
        """将数据库模型转换为业务结果对象（无 self，静态方法）。"""
        return AfterSaleInfo(
            after_sale_id=aft.after_sale_id,
            order_id=aft.order_id,
            after_sale_type=aft.after_sale_type,
            after_sale_status=aft.after_sale_status,
            refund_status=aft.refund_status,
            refund_amount=aft.refund_amount,
            created_at=aft.created_at,
            updated_at=aft.updated_at,
            reason=aft.reason,
        )
