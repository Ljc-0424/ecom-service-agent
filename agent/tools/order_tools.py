"""业务查询 Tool：将 Service 能力以受控接口提供给 Agent。"""

from __future__ import annotations

from langchain_core.tools import tool

from service.order.order_service import OrderService


@tool
def get_user_orders(user_id: str = "") -> dict:
    """查询当前用户名下的订单列表摘要。

    当用户提到「我买的 / 我的订单 / 我的商品」等却未给出明确订单号时，
    必须先调用本工具获取用户的订单列表，再定位或向用户确认是哪一笔。

    Args:
        user_id: 系统自动注入，无需填写
    """
    # 身份由 Tool Node 注入，Service 仍负责订单归属校验。
    service = OrderService()
    orders = service.list_user_orders(user_id)
    return {
        "success": True,
        "user_id": user_id,
        "order_count": len(orders),
        "orders": [o.to_dict() for o in orders],
    }


@tool
def get_order(order_id: str, user_id: str = "") -> dict:
    """根据订单号查询订单详情。

    Args:
        order_id: 订单号，从用户问题或对话上下文中获得
        user_id: 系统自动注入，无需填写
    """
    service = OrderService()
    info = service.get_order(order_id=order_id, user_id=user_id)
    if info is None:
        # 查不到包括「别人的订单」：越权在 Service 层被拦下，统一返回 not_found
        return {"success": False, "code": "not_found", "message": f"未找到订单 {order_id}"}
    return {"success": True, "order": info.to_dict()}
