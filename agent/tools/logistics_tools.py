"""物流 Tool：实时查物流（不要信历史聊天里的「已发货」）。

【官方对照】@tool + Args docstring，同 order_tools.py 文件头说明。
【名字拆解】get_logistics = get（获取）+ logistics（物流）。
"""

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
