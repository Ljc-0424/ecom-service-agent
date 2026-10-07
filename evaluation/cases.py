"""评测用例集：每条 case 描述「输入 + 应该发生的可验证事实」。

【为什么这么设计（企业最小评测闭环）】
  每条 case 只断言三类「确定性可验证」的东西：
    1. must_tools   —— 必须出现的工具（Agent 是否调对了能力）
    2. tool_args    —— 某工具的关键参数（Agent 是否把参数填对）
    3. state        —— 运行后的 State（orders_context / active_order_id / handoff）
  最终回答的文字质量**不自动断言**（自然语言没有唯一正确答案），
  评测报告里会保存每个 case 的最终回答供人工抽查。

  这套东西对应的正是需求文档里的检查维度：
  「是否调用正确 Tool / 参数是否正确 / State 是否正确 / 该 Handoff 时是否 Handoff」。

【case 覆盖的场景】
  种子数据 5 类业务场景（查列表/指定订单/库存/售后/知识）+ 4 类边界
  （越权、无效单号、多订单消歧、转人工）+ 1 类不应调工具的闲聊。

【注意】
  用例**禁止依赖**种子数据里的固定 ID 之外的信息（如具体商品名）；
  可以引用 A001 这类种子 ID——它们是 Mock DB 的公开初始状态。
"""

from __future__ import annotations

# 全部业务工具名（must_not 闲聊用例时列出，防止模型乱调工具）
_ALL_BIZ_TOOLS = [
    "get_user_orders",
    "get_order",
    "get_inventory",
    "get_logistics",
    "get_after_sale",
    "search_knowledge_base",
]

CASES: list[dict] = [
    {
        "name": "查订单列表",
        "user_id": "U001",
        "message": "我有哪些订单？",
        "must_tools": ["get_user_orders"],
        "tool_args": {},
        "state": {"orders_context_contains": ["A001", "A002"]},
    },
    {
        "name": "指定订单查物流",
        "user_id": "U001",
        "message": "A002 的快递到哪了？",
        "must_tools": ["get_logistics"],
        "tool_args": {},
        # 注意：V1 设计中物流查询结果不合并进 orders_context（只有订单查询会），
        # 所以这里不断言 orders_context
        "state": {},
    },
    {
        "name": "库存查询",
        "user_id": "U001",
        "message": "Redmi K80 还有货吗？",
        "must_tools": ["get_inventory"],
        "tool_args": {"get_inventory": {"product_name": "Redmi K80"}},
        "state": {},
    },
    {
        "name": "售后进度查询",
        "user_id": "U001",
        "message": "我 AirPods 那一单退款退到哪一步了？",
        "must_tools": ["get_after_sale"],
        "tool_args": {},
        "state": {"orders_context_contains": ["A003"]},
    },
    {
        "name": "知识检索_优惠券退款",
        "user_id": "U001",
        "message": "用优惠券买的手机，还能申请退款吗？",
        "must_tools": ["search_knowledge_base"],
        "tool_args": {},
        "must_not_tools": ["handoff_to_human"],
        "state": {},
    },
    {
        "name": "越权拦截_查他人订单",
        "user_id": "U002",  # A001 属于 U001，U002 无权查询
        "message": "帮我查一下订单 A001 的物流",
        "must_tools": [],
        # 模型可能走 get_order（订单归属校验）或 get_logistics（物流归属校验），
        # 两条路径都必须拿不到数据（Service 层校验）
        "must_tools_any": [["get_order", "get_logistics"]],
        "tool_args": {},
        "state": {"orders_context_excludes": ["A001"], "handoff": False},
    },
    {
        "name": "无效订单号",
        "user_id": "U001",
        "message": "帮我查一下订单 X99999",
        "must_tools": ["get_order"],
        "tool_args": {"get_order": {"order_id": "X99999"}},
        "state": {"orders_context_excludes": ["X99999"], "handoff": False},
    },
    {
        "name": "多订单模糊消歧",
        "user_id": "U001",
        "message": "我买的一部手机屏幕坏了，要走售后",
        "must_tools": [],
        "must_tools_any": [["get_user_orders", "get_order"]],
        "tool_args": {},
        # 两部手机 A001/A002 都应进入上下文，且 LLM 无权设置当前订单
        "state": {
            "orders_context_contains": ["A001", "A002"],
            "active_order_id_none": True,
        },
    },
    {
        "name": "转人工",
        "user_id": "U001",
        "message": "我等不了了，直接给我转人工客服，别让机器人回了",
        "must_tools": [],
        "must_tools_any": [["handoff_to_human"]],
        "tool_args": {},
        "state": {"handoff": True},
    },
    {
        "name": "闲聊不调工具",
        "user_id": "U001",
        "message": "你是谁呀？你能帮我做什么？",
        "must_tools": [],
        "must_not_tools": _ALL_BIZ_TOOLS,
        "tool_args": {},
        "state": {"handoff": False},
    },
    {
        "name": "超时未发货场景",
        "user_id": "U001",
        "message": "A005 超过承诺发货时间了还没发货，我要投诉了，怎么办？",
        "must_tools": [],
        "must_tools_any": [["get_order", "get_logistics"]],
        "tool_args": {},
        "state": {"orders_context_contains": ["A005"]},
    },
    {
        "name": "组合查询_售后加物流",
        "user_id": "U001",
        "message": "A003 的退款钱什么时候到账？顺便看看它的物流到哪了。",
        "must_tools": ["get_after_sale", "get_logistics"],
        "tool_args": {},
        "state": {},
    },
]
