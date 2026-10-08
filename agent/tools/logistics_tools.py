"""物流查询 Tool：使用当前物流数据，不依赖历史对话中的状态描述。"""

from __future__ import annotations

from langchain_core.tools import tool

from service.logistics.logistics_service import LogisticsService


@tool
def get_logistics(order_id: str = "", tracking_number: str = "", user_id: str = "") -> dict:
    """查询订单或运单号对应的最新物流状态。

    Args:
        order_id: 订单号
        tracking_number: 运单号（可选，优先使用）
        user_id: 系统自动注入，无需填写
    """
    service = LogisticsService()
    # 「x or None」：空字符串统一转 None，Service 端好判断「参数有没有传」；
    # user_id 传入后 Service 会校验订单归属，查别人的物流一律返回查无结果
    info = service.get_for_order_context(
        order_id=order_id or None,
        tracking_number=tracking_number or None,
        user_id=user_id or None,
    )
    if info is None:
        return {
            "success": False,
            "code": "not_found",
            "message": "暂无物流信息（可能尚未发货）",
        }
    return {
        "success": True,
        "logistics": info.to_dict(),
        "needs_human_attention": service.needs_human_attention(info),
    }
