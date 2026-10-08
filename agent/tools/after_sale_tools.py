"""售后查询 Tool；仅查询状态，不执行退款等敏感操作。"""

from __future__ import annotations

from langchain_core.tools import tool

from service.after_sale.after_sale_service import AfterSaleService


@tool
def get_after_sale(order_id: str = "", after_sale_id: str = "", user_id: str = "") -> dict:
    """查询售后/退款状态。

    Args:
        order_id: 订单号（可选）
        after_sale_id: 售后单号（可选，与 order_id 至少提供一个）
        user_id: 系统自动注入，无需填写
    """
    service = AfterSaleService()
    # Service 校验售后记录与用户的归属关系。
    info = service.get_after_sale(
        after_sale_id=after_sale_id or None,
        order_id=order_id or None,
        user_id=user_id or None,
    )
    if info is None:
        return {"success": False, "code": "not_found", "message": "未找到售后记录"}
    return {"success": True, "after_sale": info.to_dict()}
