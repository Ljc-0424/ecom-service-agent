# 电商客服 Agent · 第一版

> 电商智能客服 Agent + 模拟电商业务系统 + 模拟业务控制台 + 评测预留  
> 技术栈：**Python 3.11 / LangGraph / FastAPI / SQLite / Vue 3(CDN)**

## 项目定位

```text
用户问题
   ↓
Agent 理解问题
   ↓
判断需要什么信息
   ├── 企业知识 → RAG
   └── 实时业务数据 → Business Tool
   ↓
综合判断
   ↓
能够可靠回答 → 回复用户
无法可靠判断 → Human Handoff
```

核心原则：**Agent 可以基于企业规则和真实业务数据进行判断，但不能自行创造企业规则。**

第一版不执行：修改地址、直接退款、取消订单、修改规格、换货、资金操作。

---

## 目录结构

```text
ecom-service-agent/
├── agent/
│   ├── graph/          # LangGraph Workflow（LLM Node / Tool Node / Handoff / should_continue）
│   ├── state/          # AgentState：messages / orders_context / active_order_id
│   ├── runtime/        # RuntimeContext：user_id（入口显式传入）
│   ├── llm/            # LLM 封装（真实 Tool Calling；测试用 Stub）
│   └── tools/          # 业务 Tool 适配层
├── service/            # 业务逻辑（不依赖 LLM）
│   ├── order/ product/ inventory/ logistics/ after_sale/ rag/
├── db/
│   ├── business/       # 模拟业务数据库（SQLite）
│   ├── vector/         # 知识库轻量向量存储
│   └── session/        # 会话 / HandoffContext
├── api/                # FastAPI：/api/chat、/api/orders、/api/mock/*
├── frontend/           # Vue 3 单页：聊天 + 订单 + 模拟业务控制台
├── knowledge/          # 企业知识 Markdown（退换货/发货/售后/商品/客服规范）
├── evaluation/         # 评测预留：datasets / cases / runner / metrics / reports
├── tests/              # 基础测试（Agent Loop / State / 确定性规则）
├── config/             # 环境变量配置
├── data/               # SQLite / 向量 / 会话落盘目录
└── .env.example
```

---

## 架构边界

```text
                Vue Frontend
                     │
     ┌───────────────┴───────────────┐
     ↓                               ↓
Chat Interface                 Mock Business Console
     ↓                               ↓
 /api/chat                      /api/mock/*
     ↓                               ↓
   Agent                           Service
     ↓                               ↓
   Tools ──────────────────────→ Business DB
     │
     ↓
    RAG
     ↓
 Vector Store

Evaluation Layer（预留）← Trace ← Agent Run
```

| 模块 | 职责 |
|------|------|
| LLM | 理解、判断、决策 |
| Graph | 工作流控制 |
| State | Agent 当前结构化工作状态 |
| Runtime Context | 请求级身份（user_id） |
| Tool | Agent 能力入口 |
| Service | 业务逻辑 + 确定性规则 |
| Business DB | 实时业务事实 |
| RAG / Vector Store | 企业知识检索 |
| Human Handoff | 无法可靠自动处理时的人工作业入口 |
| Mock Console | 业务状态模拟器 / Agent 测试环境 |
| Evaluation | 数据集、Trace、指标（预留） |

---

## 快速开始

### 1. 安装依赖

```bash
cd ecom-service-agent
pip install fastapi uvicorn pydantic pydantic-settings langgraph langchain-core langchain-openai httpx
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写（**真实运行必须配置 LLM_API_KEY**）：

```bash
cp .env.example .env
```

无 API Key 时 `/api/chat` 会明确报错，不会用关键词假决策冒充 Tool Calling。
单元测试使用 `ScriptedStubLLM`（按脚本返回 tool_calls）。

关键项：

| 变量 | 说明 |
|------|------|
| `LLM_API_KEY` | OpenAI 兼容 API Key |
| `LLM_BASE_URL` | API Base URL |
| `LLM_MODEL` | 模型名 |

### 3. 启动服务

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

浏览器打开：<http://127.0.0.1:8000/>

- 左侧：客服聊天
- 右上：当前用户订单（点击可切换 active_order_id）
- 右下：模拟业务控制台（下单 / 改状态 / 模拟发货 / 售后 / 重置）

### 4. 跑测试

```bash
python -m unittest tests.test_agent_core -v
```

---

## 核心业务场景

| 场景 | 示例问题 | 路径 |
|------|----------|------|
| 商品知识 | 这个手机支持无线充电吗？ | RAG |
| 库存 | 这个商品有现货吗？ | Inventory Tool |
| 订单状态 | 我昨天买的东西发货了吗？ | Order Tool |
| 物流 | 我的快递到哪里了？ | Logistics Tool |
| 订单+规则 | 用了优惠券还能退款吗？ | Order Tool + RAG |
| 售后状态 | 退款怎么还没到账？ | AfterSale Tool |
| 售后政策 | 商品坏了怎么办？ | RAG |
| 订单异常 | 订单为什么被取消？ | Order Tool |
| 超时发货 | 超过承诺发货时间还没发货？ | Order + Service 规则 + 必要时 Handoff |

---

## 多轮 / 多订单

- `active_order_id`：当前讨论焦点订单（用户选择或明确意图后设置）
- `orders_context`：对话中出现过的多个订单上下文，**互不覆盖**
- 用户切换商品提问时，重新定位对应订单
- 实时数据原则：历史 Agent 回复不是实时事实，物流/订单必须重新查询

---

## 订单消歧

查询结果存在多个候选订单时，前端展示结构化选择：

```json
{ "type": "order_selection", "order_id": "A001" }
```

事件经验证后更新 `active_order_id`，不让 LLM 猜订单号。

---

## Mock Console 与 Agent 的关系

```text
Mock Console → Mock API → Service → Business DB
LLM → Tool → Service → Business DB
```

两条路径操作**同一套**模拟业务数据。  
点击控制台改变订单状态后，Agent 下一次查询能看到变化。

Mock API **不是** Agent Tool，也不直接修改 Agent State。

---

## 评测预留

```text
evaluation/
├── datasets/sample_cases.json   # 示例用例（含 initial_state / expected）
├── cases/case_schema.py         # 用例结构
├── runner/runner.py             # Reset → Run → Trace → Evaluate
├── metrics/metrics.py           # 过程指标 + 答案指标
└── reports/                     # 报告输出
```

原则：

- 模拟业务数据可控、可重置、可观察
- Agent 执行过程保留结构化 Trace
- 评测不依赖前端页面
- 不把测试数据硬编码在 Tool 里

---

## 第一版明确不做

Multi-Agent、MCP、A2A、复杂 Middleware、自动退款/取消/改地址、真实支付、复杂权限、大规模并发优化、复杂 Retry/Recovery。

原则：**先实现普通情况 → 运行 → 暴露真实问题 → 再引入对应工程能力。**

---

## 实现阶段对照

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | Agent Core（State/Runtime/LLM/Graph/Loop） | ✅ |
| Phase 2 | 业务 Tool + OrderService + 模拟 DB | ✅ |
| Phase 3 | orders_context / active_order_id / 消歧 | ✅ |
| Phase 4 | Inventory / Logistics / AfterSale / Product | ✅ |
| Phase 5 | RAG 知识库检索 | ✅ |
| Phase 6 | 综合业务问题（订单+规则） | ✅ |
| Phase 7 | Human Handoff + HandoffContext | ✅ |
| 附加 | Mock Console + 前端 + 评测预留 | ✅ |
