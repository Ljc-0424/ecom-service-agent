"""库存业务服务。第一版只查询，不修改库存。

【为什么只读】
  库存扣减涉及下单/支付等写流程，V1 的 Agent 是「纯客服」没有交易权限；
  库存变更由 Mock Console（api/mock.py）模拟完成。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from db.business.database import BusinessDatabase, get_business_db
from service.product.product_service import ProductService


@dataclass
class InventoryInfo:
    """库存业务结果对象：附上商品名和状态文字，方便直接给用户看。"""

    product_id: str  # 商品编号
    product_name: str  # 商品名（查不到商品时回退显示编号）
    available_stock: int  # 当前可售库存数量
    stock_status: str  # 库存状态：无货 / 库存紧张 / 有货（按数量现算）

    def to_dict(self) -> dict[str, Any]:
        """转成字典，方便塞进 ToolMessage / JSON 响应。"""
        return asdict(self)


class InventoryService:
    """库存业务逻辑。"""

    def __init__(self, db: Optional[BusinessDatabase] = None) -> None:
        """初始化：默认用全局业务库；同时准备商品服务（名称→ID 要用）。"""
        self.db = db or get_business_db()
        self.product_service = ProductService(self.db)

    def get_inventory(
        self,
        product_id: Optional[str] = None,
        product_name: Optional[str] = None,
    ) -> Optional[InventoryInfo]:
        """查询商品库存，支持按 ID 或名称查。

        【流程】
          1. 只有名称没有 ID → 先按名称查到商品，换出 product_id
             （用户说「iPhone 16 还有货吗」，模型只会给名称）
          2. 仍然没有 ID → 查不到，返回 None
          3. 按 ID 查库存行 → 连商品名一起组装成 InventoryInfo
        """
        if not product_id and product_name:
            product = self.product_service.find_by_name(product_name)
            if product:
                product_id = product.product_id
        if not product_id:
            return None

        inv = self.db.get_inventory(product_id)
        if inv is None:
            return None
        product = self.product_service.get_product(product_id)
        return InventoryInfo(
            product_id=product_id,
            product_name=product.product_name if product else product_id,
            available_stock=inv.available_stock,
            stock_status=inv.stock_status,
        )
