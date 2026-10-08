# Evaluation 目录

这里保留一个**最小真实模型回归脚本**，用于验证 Agent 是否选择了正确工具、
传入了关键参数，以及是否维护了预期 State。它不是完整评测平台，也不把自然语言
回答粗暴地转换成一个“准确率”数字。

运行前需要配置真实模型：

```bash
python -m evaluation.run_eval
python -m evaluation.run_eval --case 转人工
```

脚本使用独立评测数据库，并在每条 case 前重置种子数据，报告只保存确定性断言结果
和最终回答，便于人工抽查。测试管线本身使用 `ScriptedStubLLM`，不代表真实模型能力。

后续如果要扩展，应优先增加真实业务 case 和多轮稳定性统计，暂不引入 LLM-as-a-Judge
或复杂评测平台。
