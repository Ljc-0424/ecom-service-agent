"""业务数据库领域模型（模拟电商系统）。

【这个文件在干什么】
  两类东西：
  1. 状态常量类（OrderStatus 等）—— 集中定义业务状态的字符串值
  2. @dataclass 模型类（Order 等）—— 和数据库表一行一行的数据对应

  上层（Service / Tool / API）传数据都用这些对象，不直接碰 SQL 的行。

字段按第一版 MVP 场景倒推，不堆无用字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---- 业务状态常量 ----
# 为什么用「只装常量的类」：代码里写 OrderStatus.PAID 而不是裸写 "待发货"——
# 拼错立刻能发现（IDE/测试会报错），改文案也只改这一处。
# V1 用类常量够用；以后需要校验/遍历时再换成 Enum。
# 命名规律：大写常量 = 英文标识，值 = 给用户看的中文状态。

class OrderStatus:
    """订单状态流转：待支付 → 待发货 → 已发货 → 已签收；分支：已取消 / 退款中 → 已退款。"""

    PENDING_PAYMENT = "待支付"
    PAID = "待发货"  # 注意：PAID 指「已付款待发货」，不是「已完成」
    SHIPPED = "已发货"
    DELIVERED = "已签收"
    CANCELLED = "已取消"
    REFUNDING = "退款中"
    REFUNDED = "已退款"


class PaymentStatus:
    """支付状态：未支付 → 已支付；退款成功 → 已退款。"""

    UNPAID = "未支付"
    PAID = "已支付"
    REFUNDED = "已退款"


class LogisticsStatus:
    """物流状态：未发货 → 运输中 → 派送中 → 已签收；分支：运输异常 / 退回。"""

    NOT_SHIPPED = "未发货"
    SHIPPING = "运输中"
    DELIVERING = "派送中"
    DELIVERED = "已签收"
    EXCEPTION = "运输异常"
    RETURNED = "退回"


class AfterSaleStatus:
    """售后处理状态：待处理 → 处理中 → 已完成 / 已拒绝。"""

    PENDING = "待处理"
    PROCESSING = "处理中"
    COMPLETED = "已完成"
    REJECTED = "已拒绝"


class RefundStatus:
    """退款进度：无 → 退款处理中 → 退款成功 / 退款失败。"""

    NONE = "无"
    PROCESSING = "退款处理中"
    SUCCESS = "退款成功"
    FAILED = "退款失败"


class AfterSaleType:
    """售后类型：退款 / 退货 / 换货 / 维修。"""

    REFUND = "退款"
    RETURN = "退货"
    EXCHANGE = "换货"
    REPAIR = "维修"


@dataclass
class User:
    """平台用户（对应 users 表一行）。"""

    user_id: str  # 用户编号（如 U001）
    username: str  # 用户名（如「张三」）
    phone: str = ""  # 手机号（脱敏存储，如 138****0001）


@dataclass
class Product:
    """商品（对应 products 表一行）。description/usage 用于回答商品咨询。"""

    product_id: str  # 商品编号（如 P001）
    product_name: str  # 商品名
    description: str = ""  # 商品简介
    specifications: str = ""  # 规格（尺寸 / 容量 / 颜色）
    usage: str = ""  # 使用说明 / 卖点
    price: float = 0.0  # 单价（元）
    sku: str = ""  # 库存量单位编号（供应商侧编码）


@dataclass
class Inventory:
    """库存（对应 inventory 表一行）。"""

    product_id: str  # 商品编号
    available_stock: int = 0  # 当前可售库存数量

    @property
    def stock_status(self) -> str:
        """根据可售库存推导库存状态（无货/紧张/有货）。

        【为什么写成方法而不是存字段】
          阈值规则可能要调；每次现算，永远和 available_stock 一致，
          不会出现「库存改了、状态忘了改」的不同步。
        """
        if self.available_stock <= 0:
            return "无货"
        if self.available_stock <= 5:
            return "库存紧张"
        return "有货"


@dataclass
class Order:
    """订单（对应 orders 表一行）。

    关键字段：
      order_status / payment_status —— 取值见上面的常量类
      promised_ship_time            —— 承诺发货时间，超时判断依据
      tracking_number / after_sale_id —— 关联物流单 / 售后单
    """



    order_id: str  # 订单号（如 A001）
    user_id: str  # 下单用户编号
    product_id: str  # 商品编号
    product_name: str  # 商品名（下单时冗余一份，方便直接展示）
    quantity: int = 1  # 数量
    specification: str = ""  # 规格
    amount: float = 0.0  # 金额（单价 × 数量）
    created_at: str = ""  # 下单时间
    payment_status: str = PaymentStatus.UNPAID  # 支付状态（默认未支付）
    order_status: str = OrderStatus.PENDING_PAYMENT  # 订单状态（默认待支付）
    promised_ship_time: str = ""  # 承诺发货时间（超时判断依据）
    shipped_at: Optional[str] = None  # 实际发货时间（未发货为 None）
    tracking_number: Optional[str] = None  # 运单号（未发货为 None）
    after_sale_id: Optional[str] = None  # 关联售后单号（没有为 None）
    address: str = ""  # 收货地址
    cancel_reason: Optional[str] = None  # 取消原因（未取消为 None）


@dataclass
class Logistics:
    """物流运单（对应 logistics 表一行），记录运输全过程。"""

    tracking_number: str  # 快递运单号（主键，如 SF123456）
    order_id: str = ""  # 关联的订单号
    logistics_status: str = LogisticsStatus.NOT_SHIPPED  # 物流状态（默认未发货）
    current_location: str = ""  # 当前位置（如「北京分拣中心」）
    estimated_time: str = ""  # 预计送达时间
    shipped_at: Optional[str] = None  # 发货时间
    arrived_at: Optional[str] = None  # 到达目的地时间
    signed_at: Optional[str] = None  # 签收时间
    sign_info: str = ""  # 签收信息（如「本人签收」）
    is_exception: bool = False  # 是否运输异常
    exception_reason: str = ""  # 异常原因（正常为空）


@dataclass
class AfterSale:
    """售后单（对应 after_sales 表一行）。"""

    after_sale_id: str  # 售后单号（如 AS001）
    order_id: str  # 关联的订单号
    after_sale_type: str = AfterSaleType.REFUND  # 售后类型（默认退款）
    after_sale_status: str = AfterSaleStatus.PENDING  # 处理进度（默认待处理）
    refund_status: str = RefundStatus.NONE  # 退款进度（默认无）
    refund_amount: float = 0.0  # 退款金额
    created_at: str = ""  # 创建时间
    updated_at: str = ""  # 最近更新时间
    reason: str = ""  # 售后原因（用户填写）


@dataclass
class BusinessResult:
    """Tool / API 统一返回的业务结果包装。

    【V1 状态】目前是预留结构：实际代码里各接口用 dict 直接返回
    {success, code, message, ...}。等接口多了、格式需要强约束时再启用。
    """

    success: bool  # 是否成功
    code: str = "ok"  # 结果码（如 not_found / tool_error）
    message: str = ""  # 说明文字
    data: object = None  # 业务数据载荷
