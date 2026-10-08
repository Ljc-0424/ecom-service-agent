# Evaluation 后续边界

## 当前已实现：最小真实模型回归

Agent 运行应保留最小 Trace：

- User Input
- AIMessage
- Tool Call
- Tool Arguments
- Tool Result
- State Change（如 order_selection → active_order_id）
- Final Answer
- handoff status / reason

`evaluation/run_eval.py` 基于这些 Trace 做 12 条小样本 case 的确定性断言，
用于开发回归，不代表真实生产质量指标。

## [POST-V1] 后续评测方向

- Tool Selection Evaluation
- Tool Argument Evaluation
- State Evaluation
- RAG Evaluation
- Final Answer Evaluation
- End-to-End Evaluation

## 当前不做

- 自动评分 / LLM-as-a-Judge
- Benchmark / Dataset 管理平台
- 大规模回归测试平台
- 多指标评分系统

后续扩展评测时：直接基于 Trace 与会话 events，不要回过头改 Agent 核心去适配评测。
