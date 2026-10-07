"""评测运行器：真实模型跑一遍全部 case，输出逐条结果与通过率报告。

【怎么跑】（需要 .env 配好 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL）
      python -m evaluation.run_eval                 # 全部 case
      python -m evaluation.run_eval --case 转人工    # 只跑指定 case（调试用）

【为什么不做「假模型评测模式」】
  pytest 单元测试（ScriptedStubLLM）已经负责无 API 的回归验证；
  评测 runner 的职责是评**真实模型的决策质量**——工具选没选对、参数填没填对、
  State 有没有被污染。假模型按剧本演出 Expected 行为再打分是自欺欺人。

【隔离与可重复（需求文档：可控、可重置、可观察）】
  评测用独立数据库 data/eval_ecom.db（不碰演示库），
  每条 case 跑之前 reset 成种子状态，起点永远一致。

【产出】
  控制台逐条 PASS/FAIL + 汇总通过率（退出码：全过 0，有失败 2）；
  JSON 报告写入 evaluation/reports/，含每条 case 的最终回答，供人工抽查文字质量。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph.workflow import extract_final_answer, run_agent  # noqa: E402
from agent.llm.client import create_llm  # noqa: E402
from agent.state.state import empty_state  # noqa: E402
from db.business.database import BusinessDatabase, set_business_db  # noqa: E402
from db.business.seed import seed_demo_data  # noqa: E402
from db.vector.store import VectorStore  # noqa: E402
from evaluation.cases import CASES  # noqa: E402


def run_one_case(case: dict, llm, eval_db: BusinessDatabase) -> dict:
    """跑单条 case：重置业务库 → 执行 → 从结果提取「实际发生的事」→ 断言。

    返回 {name, passed, checks, actual_tools, final_answer}。
    """
    # 每条 case 前重置为种子状态（可重复性的关键）
    eval_db.reset_and_seed(seed_demo_data)

    result = run_agent(
        user_message=case["message"],
        user_id=case["user_id"],
        state=empty_state(),
        llm=llm,
    )

    # ---- 提取实际发生的事 ----
    # trace 里 type=="tool" 的条目就是每次工具调用（名字 + 系统注入后的参数）
    tool_calls = [t for t in (result.get("trace") or []) if t.get("type") == "tool"]
    actual_tools = [t["name"] for t in tool_calls]
    actual_args: dict[str, dict] = {}
    for t in tool_calls:
        actual_args.setdefault(t["name"], {}).update(t.get("args") or {})

    # handoff_to_human 走条件边直达转人工节点、不经过工具节点，
    # 不会出现在 trace 的 tool 记录里——只要触发了转人工就补记这一次调用
    if result.get("handoff") and "handoff_to_human" not in actual_tools:
        actual_tools.append("handoff_to_human")

    orders_ctx = result.get("orders_context") or {}
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # ---- 断言 1：必须出现的工具 ----
    for tool in case.get("must_tools") or []:
        check(f"必须调用 {tool}", tool in actual_tools, f"实际调用: {actual_tools}")

    # ---- 断言 2：候选组里至少调用一个（给「模型二选一都算对」的弹性） ----
    for group in case.get("must_tools_any") or []:
        check(
            f"至少调用 {' / '.join(group)} 之一",
            any(t in actual_tools for t in group),
            f"实际调用: {actual_tools}",
        )

    # ---- 断言 3：不该出现的工具（如闲聊 case 不许调业务工具） ----
    for tool in case.get("must_not_tools") or []:
        check(f"不应调用 {tool}", tool not in actual_tools, f"实际调用: {actual_tools}")

    # ---- 断言 4：工具参数（子集匹配：要求的键值对必须出现在实际参数里） ----
    for tool, want_args in (case.get("tool_args") or {}).items():
        got = actual_args.get(tool, {})
        for k, v in want_args.items():
            check(f"{tool} 参数 {k}={v}", got.get(k) == v, f"实际参数: {got}")

    # ---- 断言 5：State ----
    state_spec = case.get("state") or {}
    ctx_keys = list(orders_ctx.keys())
    for oid in state_spec.get("orders_context_contains") or []:
        check(f"订单上下文包含 {oid}", oid in ctx_keys, f"实际: {ctx_keys}")
    for oid in state_spec.get("orders_context_excludes") or []:
        check(f"订单上下文不含 {oid}", oid not in ctx_keys, f"实际: {ctx_keys}")
    if state_spec.get("active_order_id_none"):
        # 不变量：LLM 无权设置当前订单（只能由用户点选事件更新）
        check(
            "active_order_id 未被 LLM 修改",
            result.get("active_order_id") is None,
            f"实际: {result.get('active_order_id')}",
        )
    if "handoff" in state_spec:
        want = bool(state_spec["handoff"])
        got = bool(result.get("handoff"))
        check("handoff 状态", got == want, f"期望 {want}，实际 {got}")

    return {
        "name": case["name"],
        "passed": all(c["ok"] for c in checks),
        "checks": checks,
        "actual_tools": actual_tools,
        "final_answer": extract_final_answer(result),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="电商客服 Agent 最小评测（真实模型）")
    parser.add_argument("--case", action="append", default=[], help="只跑指定名称的 case（可多次）")
    args = parser.parse_args()

    # 评测专用数据库，替换全局单例；所有工具经 get_business_db() 拿到它
    eval_db = BusinessDatabase(db_path=PROJECT_ROOT / "data" / "eval_ecom.db")
    set_business_db(eval_db)

    # 知识库索引（首次从 knowledge/ 构建，之后读缓存）
    VectorStore.get_default()

    # 真实模型；没配 Key 在这里给出明确报错
    try:
        llm = create_llm()
    except RuntimeError as exc:
        print(f"[无法启动评测] {exc}")
        return 1

    cases = [c for c in CASES if not args.case or c["name"] in args.case]
    if not cases:
        print("没有匹配的 case，可用名称：")
        for c in CASES:
            print(" -", c["name"])
        return 1

    print(f"评测开始：共 {len(cases)} 条 case\n")

    import time

    results = []
    for case in cases:
        # 单条 case 独立运行 + 限流退避重试：一条失败/限流不拖垮整个 run
        # （免费档 API 常见 429；企业评测脚本的标准做法）
        r = None
        for attempt in range(1, 4):
            try:
                r = run_one_case(case, llm, eval_db)
                break
            except Exception as exc:  # noqa: BLE001
                wait = 20 * attempt
                print(f"[重试] {case['name']} 第 {attempt} 次失败（{type(exc).__name__}: {str(exc)[:80]}），{wait}s 后重试")
                time.sleep(wait)
                if attempt == 3:
                    r = {
                        "name": case["name"],
                        "passed": False,
                        "checks": [{"check": "运行", "ok": False, "detail": str(exc)[:200]}],
                        "actual_tools": [],
                        "final_answer": "",
                    }
        results.append(r)
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['name']}  实际调用: {r['actual_tools'] or '无'}")
        if not r["passed"]:
            for c in r["checks"]:
                if not c["ok"]:
                    print(f"    ✗ {c['check']}  {c['detail']}")
        time.sleep(1.5)  # 条目间隔，降低免费档限流概率

    passed = sum(1 for r in results if r["passed"])
    print(f"\n通过率: {passed}/{len(results)}")

    reports_dir = PROJECT_ROOT / "evaluation" / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(
        json.dumps(
            {"passed": passed, "total": len(results), "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"报告已写入: {report_path}")
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
