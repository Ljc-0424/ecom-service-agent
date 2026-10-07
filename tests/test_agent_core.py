"""基础测试：验证 Tool Calling Loop、订单服务规则、State Update。

测试使用 ScriptedStubLLM（脚本 tool_calls），验证 Graph/Tool/Service/State，
不验证真实 LLM 理解能力。

【怎么跑】
  .venv/Scripts/python.exe -m pytest tests/ -q

【unittest 速查】
  class TestXxx(unittest.TestCase) —— 一个测试类；test_ 开头的方法 = 一个用例
  setUpClass   —— 整个类跑之前执行一次（放昂贵的公共准备）
  setUp        —— 每个用例「之前」都执行一次（保证用例之间互不影响）
  assertEqual(a, b) / assertIsNone(x) / assertIn(a, b) —— 断言：
                 条件不成立 → 该用例失败，测试报告会指出失败位置

【为什么这里测不了「模型聪不聪明」】
  用的是 ScriptedStubLLM：按剧本念台词，不思考。
  它验证的是「管线通不通」：Graph 连线、Tool 执行、State 更新。
  模型的真实决策质量要靠配好 API Key 后人工评测（后续评测框架的事）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 把项目根目录加进 Python 的模块搜索路径：
# 直接运行本文件时，Python 默认找不到 agent/db 这些包（它们在上一级目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# noqa: E402 = 告诉代码检查器「import 不在文件顶部是有意的」，别报警
from agent.graph.workflow import (  # noqa: E402
    apply_order_selection,
    extract_final_answer,
    run_agent,
)
from agent.llm.client import ScriptedStubLLM  # noqa: E402
from agent.runtime.context import RuntimeContext  # noqa: E402
from db.business.database import BusinessDatabase, set_business_db  # noqa: E402
from db.business.seed import seed_demo_data  # noqa: E402
from service.order.order_service import OrderService  # noqa: E402


def _empty_state() -> dict:
    """构造空 AgentState（和 agent/state/state.py 的 empty_state 一个意思）。"""
    return {
        "messages": [],
        "orders_context": {},
        "active_order_id": None,
        "handoff": None,
        "trace": [],
    }


class TestOrderService(unittest.TestCase):
    """订单服务与确定性规则。"""

    @classmethod
    def setUpClass(cls) -> None:
        """准备独立测试数据库。"""
        cls.db = BusinessDatabase(db_path=PROJECT_ROOT / "data" / "test_ecom.db")
        cls.db.reset_and_seed(seed_demo_data)
        set_business_db(cls.db)
        cls.service = OrderService(cls.db)

    def test_list_user_orders(self) -> None:
        """验证用户订单列表查询。"""
        orders = self.service.list_user_orders("U001")
        self.assertGreaterEqual(len(orders), 3)

    def test_get_order(self) -> None:
        """验证订单详情查询。"""
        info = self.service.get_order("A001", user_id="U001")
        self.assertIsNotNone(info)
        self.assertEqual(info.order_id, "A001")

    def test_permission_boundary(self) -> None:
        """验证用户只能查看自己的订单。"""
        info = self.service.get_order("A001", user_id="U002")
        self.assertIsNone(info)


class TestStateUpdate(unittest.TestCase):
    """订单选择事件与 State Update。"""

    def setUp(self) -> None:
        """每个用例独立库。"""
        self.db = BusinessDatabase(db_path=PROJECT_ROOT / "data" / "test_ecom.db")
        self.db.reset_and_seed(seed_demo_data)
        set_business_db(self.db)

    def test_order_selection_updates_active(self) -> None:
        """验证合法订单选择会更新 active_order_id。"""
        runtime = RuntimeContext(user_id="U001")
        update = apply_order_selection(_empty_state(), "A001", runtime)
        self.assertEqual(update.get("active_order_id"), "A001")
        self.assertIn("A001", update.get("orders_context", {}))

    def test_invalid_order_selection(self) -> None:
        """验证非法订单号被拒绝。"""
        runtime = RuntimeContext(user_id="U001")
        update = apply_order_selection(_empty_state(), "NOT_EXIST", runtime)
        self.assertEqual(update, {})


class TestOwnershipBoundary(unittest.TestCase):
    """越权防护回归：物流/售后查询必须校验订单归属。

    背景：真实模型评测发现「查他人订单物流」可以绕过 get_order 的归属校验
    （模型直接调 get_logistics 就能拿到别人订单的运单信息），已在 Service
    层补上归属检查——本用例保证这个漏洞不会回来。
    """

    def setUp(self) -> None:
        self.db = BusinessDatabase(db_path=PROJECT_ROOT / "data" / "test_ecom.db")
        self.db.reset_and_seed(seed_demo_data)
        set_business_db(self.db)

    def test_logistics_denied_for_other_user(self) -> None:
        from service.logistics.logistics_service import LogisticsService

        service = LogisticsService(self.db)
        # A002（运单 SF123456）属于 U001：U002 按订单号或运单号都查不到
        self.assertIsNone(service.get_for_order_context(order_id="A002", user_id="U002"))
        self.assertIsNone(service.get_for_order_context(tracking_number="SF123456", user_id="U002"))
        # 属主本人可以查到
        self.assertIsNotNone(service.get_for_order_context(order_id="A002", user_id="U001"))

    def test_after_sale_denied_for_other_user(self) -> None:
        from service.after_sale.after_sale_service import AfterSaleService

        service = AfterSaleService(self.db)
        # A003 / AS001 属于 U001
        self.assertIsNone(service.get_after_sale(order_id="A003", user_id="U002"))
        self.assertIsNone(service.get_after_sale(after_sale_id="AS001", user_id="U002"))
        # 属主本人可以查到
        self.assertIsNotNone(service.get_after_sale(order_id="A003", user_id="U001"))


class TestAgentLoop(unittest.TestCase):
    """Tool Calling Loop（Graph / Tool / Service / State）。"""

    def setUp(self) -> None:
        """准备独立测试数据库。"""
        self.db = BusinessDatabase(db_path=PROJECT_ROOT / "data" / "test_ecom.db")
        self.db.reset_and_seed(seed_demo_data)
        set_business_db(self.db)

    def test_tool_calling_loop_with_stub(self) -> None:
        """Human → LLM(tool_call) → ToolNode → ToolMessage → LLM → Answer。

        【剧本怎么读】
          第 1 步：模型「要求」调 get_user_orders（注意：是模型说要调，
                  真正执行的是 Tool Node，走的是 Service → DB 真实查询）
          第 2 步：工具结果回到模型后，模型给出最终回答
        """
        stub = ScriptedStubLLM(
            script=[
                {"tool_calls": [{"name": "get_user_orders", "args": {}}]},
                {"content": "您有若干订单，详情已查到。"},
            ]
        )
        result = run_agent(
            user_message="我有哪些订单？",
            user_id="U001",
            state=_empty_state(),
            llm=stub,
        )
        types = [getattr(m, "type", "") for m in result.get("messages") or []]
        self.assertIn("human", types)
        self.assertIn("ai", types)
        self.assertIn("tool", types)
        self.assertTrue(extract_final_answer(result))
        # Node 层根据 Tool 结果合并 orders_context
        self.assertTrue(result.get("orders_context"))

    def test_handoff_path(self) -> None:
        """验证 handoff_to_human 进入 Human Handoff。"""
        stub = ScriptedStubLLM(
            script=[
                {
                    "tool_calls": [
                        {"name": "handoff_to_human", "args": {"reason": "超出自动处理范围"}}
                    ]
                }
            ]
        )
        result = run_agent(
            user_message="帮我处理一个复杂问题",
            user_id="U001",
            state=_empty_state(),
            llm=stub,
        )
        self.assertIsNotNone(result.get("handoff"))
        self.assertEqual(result.get("handoff", {}).get("status"), "handed_off")


class TestCheckpointerMemory(unittest.TestCase):
    """Checkpointer + thread_id：多轮会话状态由检查点延续。

    【测什么】
      同一 session_id 连续两轮对话，第二轮的 State 应包含第一轮的消息——
      这就是官方 Checkpointer 替换手写会话缓存后的核心保证。
    """

    def test_multi_turn_continuity(self) -> None:
        stub = ScriptedStubLLM(
            script=[{"content": "第一轮回答"}, {"content": "第二轮回答"}]
        )
        r1 = run_agent(
            user_message="我有哪些订单？", user_id="U001", session_id="T-TEST", llm=stub
        )
        r2 = run_agent(
            user_message="刚才我问了什么？", user_id="U001", session_id="T-TEST", llm=stub
        )
        # 第二轮运行结束后，消息序列应包含第一轮的用户消息与两轮问答
        contents = [str(getattr(m, "content", "")) for m in r2.get("messages") or []]
        self.assertTrue(any("我有哪些订单" in c for c in contents), f"消息序列: {contents}")
        self.assertTrue(any("刚才我问了什么" in c for c in contents))
        self.assertIn("第一轮回答", contents)

    def test_handoff_reset_each_turn(self) -> None:
        """每轮开始时 handoff 标记应被清空，不残留上一轮的转人工状态。"""
        stub = ScriptedStubLLM(script=[{"content": "好的，正常处理。"}])
        result = run_agent(
            user_message="继续聊点别的", user_id="U001", session_id="T-TEST2", llm=stub
        )
        self.assertIsNone(result.get("handoff"))


if __name__ == "__main__":
    unittest.main()
