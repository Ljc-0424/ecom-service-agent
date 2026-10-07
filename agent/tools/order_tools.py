"""业务 Tool：把「查业务数据」暴露给 LLM（官方 @tool 写法）。

【这个文件在干什么】
  LLM 不直接写 SQL；它只能调用这里定义的工具。
  每个 @tool 函数：收参数 → 调 Service → 把业务结果还给模型。
  Tool 是薄外壳：不判断意图、不改 State、不写死业务 ID（V1 约束）。

【官方对照】（LangChain Tools 页 / Quickstart §1）
  官方写法：
      @tool
      def multiply(a: int, b: int) -> int:
          \"\"\"Multiply `a` and `b`.

          Args:
              a: First int
              b: Second int
          \"\"\"
          return a * b
  要点：函数 docstring + 参数类型标注就是给模型看的「工具说明书」，
  模型靠它决定什么时候调、怎么填参数，所以说明必须写清楚。
  教学注释写在 docstring 外面（docstring 会被喂给模型，不放学习笔记）。

【V1 工具名单（7 个，权威清单见需求文档）】
  get_user_orders / get_order / get_inventory / get_logistics /
  get_after_sale / search_knowledge_base / handoff_to_human
  注意：没有 get_product —— 审查裁定删除，商品信息由订单/库存查询覆盖。
"""

from __future__ import annotations

from langchain_core.tools import tool

from service.order.order_service import OrderService


# 名字拆解：get_user_orders = get（获取）+ user（用户的）+ orders（订单，复数=列表）
@tool
def get_user_orders(user_id: str) -> dict:
    """查询当前用户名下的订单列表摘要。

    Args:
        user_id: 当前用户 ID（由系统注入）
    """
    # user_id 虽然在说明书里写给模型看，但真实值由 Tool Node 执行前强制注入
    service = OrderService()
    orders = service.list_user_orders(user_id)
    return {
        "success": True,
        "user_id": user_id,
        "order_count": len(orders),
        "orders": [o.to_dict() for o in orders],  # 每个订单对象转成字典
    }


# 名字拆解：get_order = get（获取）+ order（订单，单数=单笔），对应上面的复数
@tool
def get_order(order_id: str, user_id: str) -> dict:
    """根据订单号查询订单详情。

    Args:
        order_id: 订单号，从用户问题或对话上下文中获得
        user_id: 当前用户 ID（由系统注入）
    """
    service = OrderService()
    info = service.get_order(order_id=order_id, user_id=user_id)
    if info is None:
        # 查不到包括「别人的订单」：越权在 Service 层被拦下，统一返回 not_found
        return {"success": False, "code": "not_found", "message": f"未找到订单 {order_id}"}
    return {"success": True, "order": info.to_dict()}
