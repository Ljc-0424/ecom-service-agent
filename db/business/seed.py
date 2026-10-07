"""Mock Business DB 初始化数据（仅模拟业务系统启动种子）。

说明：
- 允许存在测试用用户/商品/订单，便于首次启动即可联调。
- Agent / Tool / Service 的决策逻辑不得依赖这些固定 ID 或商品名。
- 运行期业务数据通过 Mock Console 创建和修改。
"""

from __future__ import annotations

from db.business.database import BusinessDatabase
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


def seed_demo_data(db: BusinessDatabase) -> None:
    """写入第一版演示数据。

    【为什么是这几笔订单】
      每一笔覆盖一个典型客服场景，评测用例会用到：
        A001 待发货        —— 正常在途
        A002 已发货+物流   —— 多订单场景（用户名下有多单，考验定位）
        A003 已签收+退款中 —— 售后查询场景
        A004 已取消        —— 异常状态回答
        A005 超时未发货    —— 超时判断规则（is_ship_overdue）的场景

    【可重复执行】
      底层全是 upsert（存在则更新），跑多少遍结果都一样，
      所以 reset 接口反复调它也没问题。
    """
    # 用户
    db.upsert_user(User(user_id="U001", username="张三", phone="138****0001"))
    db.upsert_user(User(user_id="U002", username="李四", phone="139****0002"))

    # 商品
    db.upsert_product(
        Product(
            product_id="P001",
            product_name="iPhone 16",
            description="Apple 旗舰智能手机",
            specifications="6.1英寸 / 256GB / 黑色",
            usage="支持无线充电，IP68 防水",
            price=6999.0,
            sku="IP16-256-BLK",
        )
    )
    db.upsert_product(
        Product(
            product_id="P002",
            product_name="Redmi K80",
            description="小米高性能智能手机",
            specifications="6.67英寸 / 512GB / 白色",
            usage="支持 120W 快充",
            price=2999.0,
            sku="RK80-512-WHT",
        )
    )
    db.upsert_product(
        Product(
            product_id="P003",
            product_name="AirPods Pro 2",
            description="苹果降噪耳机",
            specifications="入耳式 / 白色",
            usage="主动降噪，支持 MagSafe 充电",
            price=1899.0,
            sku="APP2-WHT",
        )
    )
    db.upsert_product(
        Product(
            product_id="P004",
            product_name="小米手环 9",
            description="智能运动手环",
            specifications="黑色 / 标准版",
            usage="心率监测，14 天续航",
            price=249.0,
            sku="MB9-BLK",
        )
    )

    # 库存
    db.upsert_inventory(Inventory(product_id="P001", available_stock=12))
    db.upsert_inventory(Inventory(product_id="P002", available_stock=3))
    db.upsert_inventory(Inventory(product_id="P003", available_stock=0))
    db.upsert_inventory(Inventory(product_id="P004", available_stock=50))

    # 订单 A001：待发货（iPhone 16）
    db.upsert_order(
        Order(
            order_id="A001",
            user_id="U001",
            product_id="P001",
            product_name="iPhone 16",
            quantity=1,
            specification="256GB / 黑色",
            amount=6999.0,
            created_at="2026-09-25 10:00:00",
            payment_status=PaymentStatus.PAID,
            order_status=OrderStatus.PAID,
            promised_ship_time="2026-09-30 18:00:00",
            address="北京市朝阳区某街道 1 号",
        )
    )

    # 订单 A002：已发货（Redmi K80）— 用于多订单场景
    db.upsert_order(
        Order(
            order_id="A002",
            user_id="U001",
            product_id="P002",
            product_name="Redmi K80",
            quantity=1,
            specification="512GB / 白色",
            amount=2999.0,
            created_at="2026-09-20 14:30:00",
            payment_status=PaymentStatus.PAID,
            order_status=OrderStatus.SHIPPED,
            promised_ship_time="2026-09-22 18:00:00",
            shipped_at="2026-09-21 09:00:00",
            tracking_number="SF123456",
            address="北京市朝阳区某街道 1 号",
        )
    )
    db.upsert_logistics(
        Logistics(
            tracking_number="SF123456",
            order_id="A002",
            logistics_status=LogisticsStatus.SHIPPING,
            current_location="北京分拣中心",
            estimated_time="2026-09-23 20:00:00",
            shipped_at="2026-09-21 09:00:00",
        )
    )

    # 订单 A003：已签收 + 退款中（AirPods）
    db.upsert_order(
        Order(
            order_id="A003",
            user_id="U001",
            product_id="P003",
            product_name="AirPods Pro 2",
            quantity=1,
            specification="白色",
            amount=1899.0,
            created_at="2026-09-10 11:00:00",
            payment_status=PaymentStatus.PAID,
            order_status=OrderStatus.REFUNDING,
            promised_ship_time="2026-09-12 18:00:00",
            shipped_at="2026-09-11 10:00:00",
            tracking_number="SF789012",
            after_sale_id="AS001",
            address="北京市朝阳区某街道 1 号",
        )
    )
    db.upsert_logistics(
        Logistics(
            tracking_number="SF789012",
            order_id="A003",
            logistics_status=LogisticsStatus.DELIVERED,
            current_location="已签收",
            shipped_at="2026-09-11 10:00:00",
            signed_at="2026-09-12 15:20:00",
            sign_info="本人签收",
        )
    )
    db.upsert_after_sale(
        AfterSale(
            after_sale_id="AS001",
            order_id="A003",
            after_sale_type=AfterSaleType.REFUND,
            after_sale_status=AfterSaleStatus.PROCESSING,
            refund_status=RefundStatus.PROCESSING,
            refund_amount=1899.0,
            created_at="2026-09-18 16:00:00",
            updated_at="2026-09-19 09:00:00",
            reason="商品不符合预期",
        )
    )

    # 订单 A004：已取消
    db.upsert_order(
        Order(
            order_id="A004",
            user_id="U001",
            product_id="P004",
            product_name="小米手环 9",
            quantity=1,
            specification="黑色",
            amount=249.0,
            created_at="2026-09-05 09:00:00",
            payment_status=PaymentStatus.UNPAID,
            order_status=OrderStatus.CANCELLED,
            promised_ship_time="2026-09-07 18:00:00",
            cancel_reason="用户主动取消",
            address="北京市朝阳区某街道 1 号",
        )
    )

    # 订单 A005：超过承诺发货时间未发货（超时场景）
    db.upsert_order(
        Order(
            order_id="A005",
            user_id="U001",
            product_id="P004",
            product_name="小米手环 9",
            quantity=2,
            specification="黑色",
            amount=498.0,
            created_at="2026-09-28 12:00:00",
            payment_status=PaymentStatus.PAID,
            order_status=OrderStatus.PAID,
            promised_ship_time="2026-09-29 18:00:00",
            address="北京市朝阳区某街道 1 号",
        )
    )
