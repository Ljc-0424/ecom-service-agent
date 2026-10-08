"""Mock Console 写入 API：通过 Service 更新 Agent 使用的同一套模拟业务数据。

这些接口仅用于项目演示与评测，不作为 Agent Tool，也不直接修改 Agent State。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from db.business.database import get_business_db
from db.business.models import (
    AfterSale,
    AfterSaleStatus,
    AfterSaleType,
    Inventory,
    Logistics,
    LogisticsStatus,
    Order,
    OrderStatus,
    PaymentStatus,
    Product,
    RefundStatus,
    User,
)
from db.business.seed import seed_demo_data

router = APIRouter()


# ---- Schemas（请求体声明）----
class CreateOrderRequest(BaseModel):
    """下单请求体。"""

    user_id: str
    product_id: str
    quantity: int = Field(default=1, ge=1)
    specification: str = ""
    address: str = ""
    order_status: str = OrderStatus.PAID
    payment_status: str = PaymentStatus.PAID
    promised_ship_time: str = ""


class UpdateOrderRequest(BaseModel):
    """改订单请求体：只改传了的字段。"""

    order_status: Optional[str] = None
    payment_status: Optional[str] = None
    promised_ship_time: Optional[str] = None
    address: Optional[str] = None
    cancel_reason: Optional[str] = None
    shipped_at: Optional[str] = None
    tracking_number: Optional[str] = None


class UpdateLogisticsRequest(BaseModel):
    """物流状态更新请求体。"""

    logistics_status: str = LogisticsStatus.SHIPPING
    current_location: str = ""
    estimated_time: str = ""
    is_exception: bool = False
    exception_reason: str = ""
    sign_info: str = ""
    shipped_at: Optional[str] = None
    signed_at: Optional[str] = None


class CreateAfterSaleRequest(BaseModel):
    """创建售后单请求体。"""

    order_id: str
    after_sale_type: str = AfterSaleType.REFUND
    after_sale_status: str = AfterSaleStatus.PROCESSING
    refund_status: str = RefundStatus.PROCESSING
    refund_amount: float = Field(default=0.0, ge=0)
    reason: str = ""


class UpdateAfterSaleRequest(BaseModel):
    """更新售后请求体：只改传了的字段。"""

    after_sale_status: Optional[str] = None
    refund_status: Optional[str] = None
    refund_amount: Optional[float] = None


class UpdateInventoryRequest(BaseModel):
    """设置库存请求体。"""

    product_id: str
    available_stock: int = Field(..., ge=0)


# ---- 写操作（Mock Console）----

@router.post("/orders")
def create_order(req: CreateOrderRequest) -> dict[str, Any]:
    """模拟下单：写入 Business DB，返回新订单号。

    【流程】
      1. 校验商品存在 → 2. 生成订单号 → 3. 算金额、组装 Order
      → 4. 写订单 → 5. 顺带建一条「未发货」物流占位 → 6. 把运单号写回订单

      第 5 步的原因：物流和订单要一一对应，下单就建好占位，
      之后改物流状态才找得到运单。
    """
    db = get_business_db()
    product = db.get_product(req.product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="商品不存在")

    count = len(db.list_orders_by_user(req.user_id)) + 100
    order_id = f"M{count:03d}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    promised = req.promised_ship_time or _default_promised()

    order = Order(
        order_id=order_id,
        user_id=req.user_id,
        product_id=product.product_id,
        product_name=product.product_name,
        quantity=req.quantity,
        specification=req.specification or product.specifications,
        amount=product.price * req.quantity,
        created_at=now,
        payment_status=req.payment_status,
        order_status=req.order_status,
        promised_ship_time=promised,
        address=req.address,
    )
    db.upsert_order(order)
    # 同步创建物流占位（未发货），便于后续修改物流状态
    logi = Logistics(
        tracking_number=f"SF{order_id}",
        order_id=order_id,
        logistics_status=LogisticsStatus.NOT_SHIPPED,
    )
    db.upsert_logistics(logi)
    order.tracking_number = logi.tracking_number
    db.upsert_order(order)
    return {"success": True, "order_id": order_id, "order": order.__dict__}


@router.patch("/orders/{order_id}")
def update_order(order_id: str, req: UpdateOrderRequest) -> dict[str, Any]:
    """模拟修改订单字段（状态/地址/承诺发货时间等）。

    【PATCH 语义：部分更新】
      req.order_status is not None 判断「这个字段传没传」——
      传了才改，没传保持原值。这和「整体替换」的 PUT 是两码事。
    """
    db = get_business_db()
    order = db.get_order(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail="订单不存在")

    if req.order_status is not None:
        order.order_status = req.order_status
    if req.payment_status is not None:
        order.payment_status = req.payment_status
    if req.promised_ship_time is not None:
        order.promised_ship_time = req.promised_ship_time
    if req.address is not None:
        order.address = req.address
    if req.cancel_reason is not None:
        order.cancel_reason = req.cancel_reason
    if req.shipped_at is not None:
        order.shipped_at = req.shipped_at
    if req.tracking_number is not None:
        order.tracking_number = req.tracking_number

    db.upsert_order(order)  # upsert：存在即更新
    return {"success": True, "order": order.__dict__}


@router.put("/logistics/{order_id}")
def update_logistics(order_id: str, req: UpdateLogisticsRequest) -> dict[str, Any]:
    """模拟物流状态变化。新建或更新该订单运单。

    【为什么后半段在改订单】
      物流状态和订单状态必须联动，不然会「物流已签收、订单还显示已发货」：
        未发货       → 订单回到 待发货
        已签收       → 订单变 已签收
        其它（运输中/派送中）→ 订单变 已发货
    """
    db = get_business_db()
    order = db.get_order(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail="订单不存在")

    tracking = order.tracking_number or f"SF{order_id}"
    # or 右侧兜底：还没有运单就现建一条占位
    existing = db.get_logistics_by_tracking(tracking) or Logistics(
        tracking_number=tracking,
        order_id=order_id,
    )

    existing.order_id = order_id
    existing.logistics_status = req.logistics_status
    existing.current_location = req.current_location or existing.current_location
    existing.estimated_time = req.estimated_time or existing.estimated_time
    existing.is_exception = req.is_exception
    existing.exception_reason = req.exception_reason
    existing.sign_info = req.sign_info or existing.sign_info
    if req.shipped_at is not None:
        existing.shipped_at = req.shipped_at
    if req.signed_at is not None:
        existing.signed_at = req.signed_at
    if existing.shipped_at is None and req.logistics_status != LogisticsStatus.NOT_SHIPPED:
        existing.shipped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db.upsert_logistics(existing)

    # 同步订单侧字段
    order.tracking_number = tracking
    if req.logistics_status == LogisticsStatus.NOT_SHIPPED:
        order.order_status = OrderStatus.PAID
        order.shipped_at = None
    elif req.logistics_status == LogisticsStatus.DELIVERED:
        order.order_status = OrderStatus.DELIVERED
        order.shipped_at = existing.shipped_at
    else:
        if order.order_status in (OrderStatus.PAID, OrderStatus.PENDING_PAYMENT):
            order.order_status = OrderStatus.SHIPPED
        order.shipped_at = existing.shipped_at
    db.upsert_order(order)

    return {"success": True, "logistics": existing.__dict__, "order_id": order_id}


@router.post("/after-sales")
def create_after_sale(req: CreateAfterSaleRequest) -> dict[str, Any]:
    """创建售后单并同步订单退款状态。

    【联动规则】
      建完售后单：订单挂上 after_sale_id；
      退款状态不是「无」→ 订单状态改为 退款中。
    """
    db = get_business_db()
    order = db.get_order(req.order_id)
    if order is None:
        raise HTTPException(status_code=404, detail="订单不存在")

    aft_id = f"AS{req.order_id}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    aft = AfterSale(
        after_sale_id=aft_id,
        order_id=req.order_id,
        after_sale_type=req.after_sale_type,
        after_sale_status=req.after_sale_status,
        refund_status=req.refund_status,
        refund_amount=req.refund_amount or order.amount,
        created_at=now,
        updated_at=now,
        reason=req.reason,
    )
    db.upsert_after_sale(aft)
    order.after_sale_id = aft_id
    if req.refund_status and req.refund_status != RefundStatus.NONE:
        order.order_status = OrderStatus.REFUNDING
    db.upsert_order(order)
    return {"success": True, "after_sale": aft.__dict__}


@router.patch("/after-sales/{after_sale_id}")
def update_after_sale(after_sale_id: str, req: UpdateAfterSaleRequest) -> dict[str, Any]:
    """更新售后/退款状态；退款成功时同步订单。

    【联动规则】
      退款成功 → 订单的支付状态变「已退款」、订单状态变「已退款」，
      这正是 Agent 查订单/售后时要拿到的最新事实。
    """
    db = get_business_db()
    aft = db.get_after_sale(after_sale_id)
    if aft is None:
        raise HTTPException(status_code=404, detail="售后不存在")

    if req.after_sale_status is not None:
        aft.after_sale_status = req.after_sale_status
    if req.refund_status is not None:
        aft.refund_status = req.refund_status
    if req.refund_amount is not None:
        aft.refund_amount = req.refund_amount
    aft.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.upsert_after_sale(aft)

    order = db.get_order(aft.order_id)
    if order and aft.refund_status == RefundStatus.SUCCESS:
        order.payment_status = PaymentStatus.REFUNDED
        order.order_status = OrderStatus.REFUNDED
        db.upsert_order(order)

    return {"success": True, "after_sale": aft.__dict__}


@router.put("/inventory")
def update_inventory(req: UpdateInventoryRequest) -> dict[str, Any]:
    """设置商品可售库存。"""
    db = get_business_db()
    inv = Inventory(product_id=req.product_id, available_stock=req.available_stock)
    db.upsert_inventory(inv)
    return {
        "success": True,
        "inventory": {
            "product_id": inv.product_id,
            "available_stock": inv.available_stock,
            "stock_status": inv.stock_status,
        },
    }


@router.get("/products")
def list_products() -> dict[str, Any]:
    """列出商品（供下单下拉选择）。"""
    db = get_business_db()
    products = []
    for p in db.list_products():
        inv = db.get_inventory(p.product_id)
        products.append(
            {
                "product_id": p.product_id,
                "product_name": p.product_name,
                "price": p.price,
                "specifications": p.specifications,
                "available_stock": inv.available_stock if inv else 0,
            }
        )
    return {"products": products}


@router.post("/reset")
def reset_demo_data() -> dict[str, Any]:
    """重置为种子数据（评测 / 手动复位）。

    调 seed_demo_data 覆盖回初始状态；配合 upsert 的幂等性，可反复执行。
    """
    db = get_business_db()
    seed_demo_data(db)
    return {"success": True, "message": "已重置为初始种子数据"}


def _default_promised() -> str:
    """返回默认承诺发货时间。

    下单时若没指定承诺时间，默认给两天后的时间，
    保证新订单不会因为示例日期过期而刚创建就被判定为超时。
    """
    promised = datetime.now() + timedelta(days=2)
    return promised.strftime("%Y-%m-%d %H:%M:%S")
