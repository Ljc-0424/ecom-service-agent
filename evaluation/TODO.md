# Evaluation 预留（V1 不实现完整评测）

## V1 只保证可观察

Agent 运行应保留最小 Trace：

- User Input
- AIMessage
- Tool Call
- Tool Arguments
- Tool Result
- State Change（如 order_selection → active_order_id）
- Final Answer
- handoff status / reason

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
- 回归测试平台
- 多指标评分系统

实现评测时：直接基于 `agent/graph/trace.py` 与会话中的 events，不要回过头改 Agent 核心去适配评测。
