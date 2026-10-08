"""SQLite 业务数据访问层，供业务 Service 读写模拟订单、商品和售后数据。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from config.settings import settings
from db.business.models import (
    AfterSale,
    Inventory,
    Logistics,
    Order,
    Product,
    User,
)

# 建表 SQL：由 executescript() 一次性执行全部语句。
# IF NOT EXISTS —— 表已存在就跳过，保证服务重复启动不报错
# PRIMARY KEY   —— 主键，同一编号只允许一行（重复写入会被 upsert 转成更新）
# FOREIGN KEY   —— 外键，标记表之间的引用关系（如订单属于哪个用户）
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    phone TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    description TEXT DEFAULT '',
    specifications TEXT DEFAULT '',
    usage TEXT DEFAULT '',
    price REAL DEFAULT 0,
    sku TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS inventory (
    product_id TEXT PRIMARY KEY,
    available_stock INTEGER DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    product_name TEXT NOT NULL,
    quantity INTEGER DEFAULT 1,
    specification TEXT DEFAULT '',
    amount REAL DEFAULT 0,
    created_at TEXT DEFAULT '',
    payment_status TEXT DEFAULT '未支付',
    order_status TEXT DEFAULT '待支付',
    promised_ship_time TEXT DEFAULT '',
    shipped_at TEXT,
    tracking_number TEXT,
    after_sale_id TEXT,
    address TEXT DEFAULT '',
    cancel_reason TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS logistics (
    tracking_number TEXT PRIMARY KEY,
    order_id TEXT DEFAULT '',
    logistics_status TEXT DEFAULT '未发货',
    current_location TEXT DEFAULT '',
    estimated_time TEXT DEFAULT '',
    shipped_at TEXT,
    arrived_at TEXT,
    signed_at TEXT,
    sign_info TEXT DEFAULT '',
    is_exception INTEGER DEFAULT 0,
    exception_reason TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS after_sales (
    after_sale_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    after_sale_type TEXT DEFAULT '退款',
    after_sale_status TEXT DEFAULT '待处理',
    refund_status TEXT DEFAULT '无',
    refund_amount REAL DEFAULT 0,
    created_at TEXT DEFAULT '',
    updated_at TEXT DEFAULT '',
    reason TEXT DEFAULT ''
);
"""


class BusinessDatabase:
    """SQLite 业务库封装（V1 单请求模型，不设并发锁）。

    【职责】
      建表 + 查询 + 写入。上层（Service）只调方法拿业务对象，
      完全不接触 SQL 和 sqlite3。

    【V1 的简化】
      每次操作都新开一个连接、用完就关。简单可靠，
      不处理多线程并发（评测场景一次只有一个人在用）。
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """初始化实例并建表。

        【参数】
          db_path —— 数据库文件路径。不传则用全局配置 settings.sqlite_path；
                     测试时传独立路径（如 test_ecom.db），不污染演示数据。

        【流程】
          记下路径 → 确保所在目录存在 → 建表（已存在就跳过）
        """
        self.db_path = Path(db_path) if db_path else settings.sqlite_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)  # data 目录不存在则建
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """创建一个新的 SQLite 连接。

        【两行关键配置】
          check_same_thread=False —— 允许跨线程共用连接（V1 单请求其实用不到）
          row_factory = sqlite3.Row —— 查询结果按「列名」取值，如 row["order_id"]；
                                       不设置的话只能按位置 row[0]，可读性差
        """
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        """提供数据库游标；正常退出时提交，异常时关闭连接。"""
        conn = self._connect()
        try:
            cur = conn.cursor()
            yield cur
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """初始化业务表结构（执行上面的建表脚本）。"""
        with self.cursor() as cur:
            cur.executescript(_SCHEMA)

    # ---- User ----

    def get_user(self, user_id: str) -> Optional[User]:
        """按用户 ID 查询用户，使用参数化 SQL 避免拼接查询语句。"""
        with self.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cur.fetchone()
        return User(**dict(row)) if row else None

    def upsert_user(self, user: User) -> None:
        """新增或更新用户记录。"""
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO users (user_id, username, phone)
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                     username = excluded.username,
                     phone = excluded.phone""",
                (user.user_id, user.username, user.phone),
            )

    # ---- Product ----

    def get_product(self, product_id: str) -> Optional[Product]:
        """按商品 ID 查询商品。"""
        with self.cursor() as cur:
            cur.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
            row = cur.fetchone()
        return Product(**dict(row)) if row else None

    def get_product_by_name(self, product_name: str) -> Optional[Product]:
        """按商品名称模糊查询商品。"""
        with self.cursor() as cur:
            cur.execute(
                "SELECT * FROM products WHERE product_name LIKE ? LIMIT 1",
                (f"%{product_name}%",),
            )
            row = cur.fetchone()
        return Product(**dict(row)) if row else None

    def list_products(self) -> list[Product]:
        """列出全部商品。"""
        with self.cursor() as cur:
            cur.execute("SELECT * FROM products")
            return [Product(**dict(r)) for r in cur.fetchall()]

    def upsert_product(self, product: Product) -> None:
        """新增或更新商品。"""
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO products
                   (product_id, product_name, description, specifications, usage, price, sku)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(product_id) DO UPDATE SET
                     product_name = excluded.product_name,
                     description = excluded.description,
                     specifications = excluded.specifications,
                     usage = excluded.usage,
                     price = excluded.price,
                     sku = excluded.sku""",
                (
                    product.product_id,
                    product.product_name,
                    product.description,
                    product.specifications,
                    product.usage,
                    product.price,
                    product.sku,
                ),
            )

    # ---- Inventory ----

    def get_inventory(self, product_id: str) -> Optional[Inventory]:
        """查询商品库存。"""
        with self.cursor() as cur:
            cur.execute("SELECT * FROM inventory WHERE product_id = ?", (product_id,))
            row = cur.fetchone()
        return Inventory(product_id=row["product_id"], available_stock=row["available_stock"]) if row else None

    def upsert_inventory(self, inv: Inventory) -> None:
        """新增或更新库存。"""
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO inventory (product_id, available_stock)
                   VALUES (?, ?)
                   ON CONFLICT(product_id) DO UPDATE SET
                     available_stock = excluded.available_stock""",
                (inv.product_id, inv.available_stock),
            )

    # ---- Order ----

    def get_order(self, order_id: str) -> Optional[Order]:
        """按订单 ID 查询订单原始记录。"""
        with self.cursor() as cur:
            cur.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
            row = cur.fetchone()
        return Order(**dict(row)) if row else None

    def list_orders_by_user(self, user_id: str) -> list[Order]:
        """按用户查询订单列表（按创建时间倒序）。"""
        with self.cursor() as cur:
            cur.execute(
                "SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
            return [Order(**dict(r)) for r in cur.fetchall()]

    def upsert_order(self, order: Order) -> None:
        """新增或更新订单。"""
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO orders (
                    order_id, user_id, product_id, product_name, quantity,
                    specification, amount, created_at, payment_status, order_status,
                    promised_ship_time, shipped_at, tracking_number, after_sale_id,
                    address, cancel_reason
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(order_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    product_id=excluded.product_id,
                    product_name=excluded.product_name,
                    quantity=excluded.quantity,
                    specification=excluded.specification,
                    amount=excluded.amount,
                    created_at=excluded.created_at,
                    payment_status=excluded.payment_status,
                    order_status=excluded.order_status,
                    promised_ship_time=excluded.promised_ship_time,
                    shipped_at=excluded.shipped_at,
                    tracking_number=excluded.tracking_number,
                    after_sale_id=excluded.after_sale_id,
                    address=excluded.address,
                    cancel_reason=excluded.cancel_reason""",
                (
                    order.order_id,
                    order.user_id,
                    order.product_id,
                    order.product_name,
                    order.quantity,
                    order.specification,
                    order.amount,
                    order.created_at,
                    order.payment_status,
                    order.order_status,
                    order.promised_ship_time,
                    order.shipped_at,
                    order.tracking_number,
                    order.after_sale_id,
                    order.address,
                    order.cancel_reason,
                ),
            )

    # ---- Logistics ----

    def get_logistics_by_tracking(self, tracking_number: str) -> Optional[Logistics]:
        """按运单号查询物流。"""
        with self.cursor() as cur:
            cur.execute(
                "SELECT * FROM logistics WHERE tracking_number = ?", (tracking_number,)
            )
            row = cur.fetchone()
        return self._row_to_logistics(row) if row else None

    def get_logistics_by_order(self, order_id: str) -> Optional[Logistics]:
        """按订单号查询物流。"""
        with self.cursor() as cur:
            cur.execute(
                "SELECT * FROM logistics WHERE order_id = ? LIMIT 1", (order_id,)
            )
            row = cur.fetchone()
        return self._row_to_logistics(row) if row else None

    @staticmethod
    def _row_to_logistics(row: sqlite3.Row) -> Logistics:
        """将数据库行转换为 Logistics 模型。"""
        d = dict(row)
        d["is_exception"] = bool(d.get("is_exception", 0))
        return Logistics(**d)

    def upsert_logistics(self, logi: Logistics) -> None:
        """新增或更新物流记录。"""
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO logistics (
                    tracking_number, order_id, logistics_status, current_location,
                    estimated_time, shipped_at, arrived_at, signed_at, sign_info,
                    is_exception, exception_reason
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(tracking_number) DO UPDATE SET
                    order_id=excluded.order_id,
                    logistics_status=excluded.logistics_status,
                    current_location=excluded.current_location,
                    estimated_time=excluded.estimated_time,
                    shipped_at=excluded.shipped_at,
                    arrived_at=excluded.arrived_at,
                    signed_at=excluded.signed_at,
                    sign_info=excluded.sign_info,
                    is_exception=excluded.is_exception,
                    exception_reason=excluded.exception_reason""",
                (
                    logi.tracking_number,
                    logi.order_id,
                    logi.logistics_status,
                    logi.current_location,
                    logi.estimated_time,
                    logi.shipped_at,
                    logi.arrived_at,
                    logi.signed_at,
                    logi.sign_info,
                    1 if logi.is_exception else 0,
                    logi.exception_reason,
                ),
            )

    # ---- AfterSale ----

    def get_after_sale(self, after_sale_id: str) -> Optional[AfterSale]:
        """按售后单号查询售后。"""
        with self.cursor() as cur:
            cur.execute(
                "SELECT * FROM after_sales WHERE after_sale_id = ?", (after_sale_id,)
            )
            row = cur.fetchone()
        return AfterSale(**dict(row)) if row else None

    def get_after_sale_by_order(self, order_id: str) -> Optional[AfterSale]:
        """按订单号查询售后。"""
        with self.cursor() as cur:
            cur.execute(
                "SELECT * FROM after_sales WHERE order_id = ? LIMIT 1", (order_id,)
            )
            row = cur.fetchone()
        return AfterSale(**dict(row)) if row else None

    def upsert_after_sale(self, aft: AfterSale) -> None:
        """新增或更新售后记录。"""
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO after_sales (
                    after_sale_id, order_id, after_sale_type, after_sale_status,
                    refund_status, refund_amount, created_at, updated_at, reason
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(after_sale_id) DO UPDATE SET
                    order_id=excluded.order_id,
                    after_sale_type=excluded.after_sale_type,
                    after_sale_status=excluded.after_sale_status,
                    refund_status=excluded.refund_status,
                    refund_amount=excluded.refund_amount,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    reason=excluded.reason""",
                (
                    aft.after_sale_id,
                    aft.order_id,
                    aft.after_sale_type,
                    aft.after_sale_status,
                    aft.refund_status,
                    aft.refund_amount,
                    aft.created_at,
                    aft.updated_at,
                    aft.reason,
                ),
            )

    # ---- 重置（评测可重复性）----

    def reset_and_seed(self, seeder: Any = None) -> None:
        """清空业务表并重新灌入种子数据，保证评测可重复。

        【为什么需要】
          评测要在相同的初始数据上跑，才能对比「改了 Agent 之后效果有没有变好」。
          评测开始前调一次，业务库就回到固定的初始状态。

        【流程】
          按顺序 DELETE 每张表（先删 after_sales/logistics 这些「子表」，
          再删 orders/products/users 这些「父表」）→ seeder 不为 None
          就调用它灌入演示数据。
        """
        with self.cursor() as cur:
            for table in (
                "after_sales",
                "logistics",
                "orders",
                "inventory",
                "products",
                "users",
            ):
                # f-string 把表名填进 DELETE 语句；表名是白名单常量，无注入风险
                cur.execute(f"DELETE FROM {table}")
        if seeder is not None:
            seeder(self)  # 回调：把灌数据的具体工作交给外部传入的函数


# 全局单例（Demo / API 共用同一套业务数据）
_business_db: Optional[BusinessDatabase] = None


def get_business_db() -> BusinessDatabase:
    """获取全局业务数据库单例，供 API、Tool 和 Service 共用。"""
    global _business_db
    if _business_db is None:
        _business_db = BusinessDatabase()
    return _business_db


def set_business_db(db: BusinessDatabase) -> None:
    """测试时注入独立数据库（替换全局单例）。

    【为什么需要】
      测试不想碰 data/ecom.db 里的演示数据：先把一个指向 test_ecom.db
      的实例塞进来，之后所有 get_business_db() 拿到的都是测试库。
    """
    global _business_db
    _business_db = db
