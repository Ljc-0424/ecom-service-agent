"""订单业务逻辑：查询、归属校验、结果组装及发货超时判断。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

from db.business.database import BusinessDatabase, get_business_db
from db.business.models import Order, OrderStatus


@dataclass
class OrderSummary:
    """列表页用的精简订单（不是整行数据库）。"""

    order_id: str  # 订单号（如 A001）
    product_name: str  # 商品名
    specification: str  # 规格
    order_status: str  # 订单状态（待支付 / 待发货 / 已发货…）
    payment_status: str  # 支付状态（未支付 / 已支付 / 已退款）
    amount: float  # 订单金额
    created_at: str  # 下单时间
    has_logistics: bool  # 是否已有物流信息（未发货算没有）
    has_after_sale: bool  # 是否挂了售后单

    def to_dict(self) -> dict[str, Any]:
        """返回可序列化的订单摘要。"""
        return asdict(self)


@dataclass
class OrderInfo:
    """订单详情业务结果。"""

    order_id: str  # 订单号
    product_id: str  # 商品编号
    product_name: str  # 商品名
    specification: str  # 规格
    quantity: int  # 数量
    amount: float  # 金额（单价 × 数量）
    created_at: str  # 下单时间
    payment_status: str  # 支付状态
    order_status: str  # 订单状态
    promised_ship_time: str  # 承诺发货时间（超时判断依据）
    shipped_at: Optional[str]  # 实际发货时间（未发货为 None）
    tracking_number: Optional[str]  # 运单号（未发货为 None）
    after_sale_id: Optional[str]  # 关联售后单号（没有为 None）
    address: str  # 收货地址
    cancel_reason: Optional[str]  # 取消原因（未取消为 None）
    is_overdue: bool  # 是否超承诺时间仍未发货（Service 现算，不让 LLM 猜）
    overdue_note: str  # 超时说明文案（未超时为空）

    def to_dict(self) -> dict[str, Any]:
        """转成字典。"""
        return asdict(self)


def _parse_time(text: str) -> Optional[datetime]:
    """把字符串时间转成 datetime，方便比较先后。

    【流程】
      依次尝试几种常见格式；都不匹配返回 None。
    """
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue  # 这个格式不对，试下一个
    return None


class OrderService:
    """订单业务逻辑。"""

    def __init__(self, db: Optional[BusinessDatabase] = None) -> None:
        """初始化：默认用全局业务库，测试时可注入独立库。"""
        self.db = db or get_business_db()

    def list_user_orders(self, user_id: str) -> list[OrderSummary]:
        """查用户名下订单摘要列表。

        【流程】
          1. 从 DB 取该用户全部订单
          2. 每条订单看有没有物流/售后
          3. 装进 OrderSummary 返回
        """
        orders = self.db.list_orders_by_user(user_id)
        result: list[OrderSummary] = []
        for o in orders:
            logistics = self.db.get_logistics_by_order(o.order_id)
            after_sale = (
                self.db.get_after_sale(o.after_sale_id)
                if o.after_sale_id
                else self.db.get_after_sale_by_order(o.order_id)
            )
            result.append(
                OrderSummary(
                    order_id=o.order_id,
                    product_name=o.product_name,
                    specification=o.specification,
                    order_status=o.order_status,
                    payment_status=o.payment_status,
                    amount=o.amount,
                    created_at=o.created_at,
                    has_logistics=logistics is not None
                    and logistics.logistics_status != "未发货",
                    has_after_sale=after_sale is not None,
                )
            )
        return result

    def get_order(self, order_id: str, user_id: str) -> Optional[OrderInfo]:
        """查单笔订单详情；不是自己的订单返回 None。

        【流程】
          1. DB 查 order_id
          2. user_id 对不上 → None（权限边界）
          3. 转成 OrderInfo（含 is_overdue）
        """
        order = self.db.get_order(order_id)
        if order is None:
            return None
        if order.user_id != user_id:
            return None
        return self._to_info(order)

    def find_orders_for_user(self, user_id: str) -> list[OrderInfo]:
        """查该用户全部订单的完整信息。"""
        return [self._to_info(o) for o in self.db.list_orders_by_user(user_id)]

    def is_ship_overdue(self, order: OrderInfo, now: Optional[datetime] = None) -> bool:
        """确定性规则：是否「超了承诺时间还没发货」。

        【流程】
          1. 订单状态不是待发货/待支付 → 不算
          2. 已经发过货 → 不算
          3. 没有承诺时间 → 不算
          4. 当前时间 > 承诺时间 → True

        【为什么要放在 Service】
          这是业务规则，不能交给 LLM 猜。
        """
        if order.order_status not in (OrderStatus.PAID, OrderStatus.PENDING_PAYMENT):
            return False
        if order.shipped_at:
            return False
        promised = _parse_time(order.promised_ship_time)
        if promised is None:
            return False
        current = now or datetime.now()
        return current > promised

    def _to_info(self, order: Order) -> OrderInfo:
        """把数据库 Order 转成业务 OrderInfo，并算 is_overdue。"""
        info = OrderInfo(
            order_id=order.order_id,
            product_id=order.product_id,
            product_name=order.product_name,
            specification=order.specification,
            quantity=order.quantity,
            amount=order.amount,
            created_at=order.created_at,
            payment_status=order.payment_status,
            order_status=order.order_status,
            promised_ship_time=order.promised_ship_time,
            shipped_at=order.shipped_at,
            tracking_number=order.tracking_number,
            after_sale_id=order.after_sale_id,
            address=order.address,
            cancel_reason=order.cancel_reason,
            is_overdue=False,
            overdue_note="",
        )
        if self.is_ship_overdue(info):
            info.is_overdue = True
            info.overdue_note = "已超过承诺发货时间且仍未发货"
        return info
