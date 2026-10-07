"""商品业务服务：商品信息的查询层。

【谁在用】
  service/inventory/inventory_service.py —— 按名称查商品换 ID
  db 层的 upsert/seed 流程 —— 不经过这里（直接写库）
  V1 没有独立的 get_product Tool：商品信息可由订单/库存查询覆盖，
  避免给 LLM 堆太多工具（工具越多，模型选错的概率越大）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

from db.business.database import BusinessDatabase, get_business_db


@dataclass
class ProductInfo:
    """商品业务结果对象（模式同 LogisticsInfo：对数据库行的稳定视图）。"""

    product_id: str  # 商品编号（如 P001）
    product_name: str  # 商品名
    description: str  # 商品简介（客服回答「这是什么」用）
    specifications: str  # 规格（尺寸 / 容量 / 颜色）
    usage: str  # 使用说明 / 卖点
    price: float  # 单价（元）
    sku: str  # 库存量单位编号（供应商侧编码）

    def to_dict(self) -> dict[str, Any]:
        """转成字典，方便塞进 ToolMessage / JSON 响应。"""
        return asdict(self)


class ProductService:
    """商品业务逻辑。"""

    def __init__(self, db: Optional[BusinessDatabase] = None) -> None:
        """初始化：默认用全局业务库，测试时可注入独立库。"""
        self.db = db or get_business_db()

    def get_product(self, product_id: str) -> Optional[ProductInfo]:
        """按商品 ID 精确查询。"""
        p = self.db.get_product(product_id)
        return self._to_info(p) if p else None

    def find_by_name(self, name: str) -> Optional[ProductInfo]:
        """按名称关键词模糊查询（取第一个命中）。"""
        p = self.db.get_product_by_name(name)
        return self._to_info(p) if p else None

    def list_products(self) -> list[ProductInfo]:
        """列出全部商品（Mock Console 下拉框用）。"""
        return [self._to_info(p) for p in self.db.list_products()]

    @staticmethod
    def _to_info(p) -> ProductInfo:
        """将数据库模型转换为业务结果对象（无 self，静态方法）。"""
        return ProductInfo(
            product_id=p.product_id,
            product_name=p.product_name,
            description=p.description,
            specifications=p.specifications,
            usage=p.usage,
            price=p.price,
            sku=p.sku,
        )
