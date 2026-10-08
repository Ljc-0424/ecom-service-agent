"""订单面板查询 API；直接读取业务数据，不经过 Agent。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from service.after_sale.after_sale_service import AfterSaleService
from service.logistics.logistics_service import LogisticsService
from service.order.order_service import OrderService

router = APIRouter()


@router.get("/orders")
def list_orders(user_id: str) -> dict[str, Any]:
    """查询指定用户的订单列表摘要。"""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 不能为空")
    service = OrderService()
    orders = service.list_user_orders(user_id)
    return {
        "user_id": user_id,
        "orders": [o.to_dict() for o in orders],
    }


@router.get("/orders/{order_id}")
def get_order_detail(order_id: str, user_id: str) -> dict[str, Any]:
    """查询订单详情，含物流与售后（若有）。

    【流程】
      1. 查订单（Service 会校验归属：不是该用户的订单查不到）
      2. 有运单号 → 先按运单号查物流；没查到再按订单号兜底
      3. 查售后单（可能没有）
      4. 三份数据打包成一个 JSON 返回，前端一次请求全拿到
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 不能为空")
    order_service = OrderService()
    info = order_service.get_order(order_id=order_id, user_id=user_id)
    if info is None:
        # 404：既保护了不存在的单号，也不会泄露「别人的订单」是否存在
        raise HTTPException(status_code=404, detail="订单不存在或无权限")

    logistics_service = LogisticsService()
    logistics = None
    if info.tracking_number:
        logistics = logistics_service.get_by_tracking(info.tracking_number)
    if logistics is None:
        logistics = logistics_service.get_by_order(order_id)  # 兜底再查一次

    after_sale = AfterSaleService().get_after_sale(
        after_sale_id=info.after_sale_id,
        order_id=order_id,
    )

    return {
        "order": info.to_dict(),
        "logistics": logistics.to_dict() if logistics else None,
        "after_sale": after_sale.to_dict() if after_sale else None,
    }
