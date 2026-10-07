"""库存 Tool。

【官方对照】@tool + Args docstring，同 order_tools.py 文件头说明。
【名字拆解】get_inventory = get（获取）+ inventory（库存）。
"""

from __future__ import annotations

from langchain_core.tools import tool

from service.inventory.inventory_service import InventoryService


@tool
def get_inventory(product_id: str = "", product_name: str = "") -> dict:
    """查询商品库存/是否有现货。

    Args:
        product_id: 商品 ID（可选）
        product_name: 商品名称关键词（可选，与 product_id 至少提供一个）
    """
    service = InventoryService()
    # 「x or None」：空字符串统一转 None，Service 端好判断「参数有没有传」
    info = service.get_inventory(
        product_id=product_id or None,
        product_name=product_name or None,
    )
    if info is None:
        return {"success": False, "code": "not_found", "message": "未找到该商品库存信息"}
    return {"success": True, "inventory": info.to_dict()}
