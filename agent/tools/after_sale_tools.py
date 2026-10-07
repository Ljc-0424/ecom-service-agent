"""售后 Tool：只查询，不自动退款。

【官方对照】@tool + Args docstring，同 order_tools.py 文件头说明。
【名字拆解】get_after_sale = get（获取）+ after_sale（售后）。
【安全边界】退款是敏感操作，V1 只读；真要退款走 handoff_to_human 转人工。
"""

from __future__ import annotations

from langchain_core.tools import tool

from service.after_sale.after_sale_service import AfterSaleService


@tool
def get_after_sale(order_id: str = "", after_sale_id: str = "") -> dict:
    """查询售后/退款状态。

    Args:
        order_id: 订单号（可选）
        after_sale_id: 售后单号（可选，与 order_id 至少提供一个）
    """
    service = AfterSaleService()
    # 「x or None」：空字符串统一转 None，Service 端好判断「参数有没有传」
    info = service.get_after_sale(
        after_sale_id=after_sale_id or None,
        order_id=order_id or None,
    )
    if info is None:
        return {"success": False, "code": "not_found", "message": "未找到售后记录"}
    return {"success": True, "after_sale": info.to_dict()}
