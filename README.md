# 电商客服 Agent：自动处理订单、退换货等客服任务

## 项目背景

构建一个能在仿真电商场景中自动处理退换货、查订单等任务的智能客服 Agent。核心挑战是让一个 4B 小模型学会**准确调用工具**（查订单、处理退货等），而不是自由文本回答。

## 仿真环境

基于开源平台 **τ-bench**，自带：
- 模拟数据库（虚拟订单、用户、商品）
- 16 个工具 API（查订单、查商品、退货、换货等）
- LLM 模拟用户
- 115 个标准测试任务
- 通过数据库最终状态哈希客观评分（PASS/FAIL）

## 结果

| 阶段 | 模型 | pass^1 | 说明 |
|---|---|---|---|
| 基线 | Qwen3-4B-Instruct (base) | **7.0%** | 小模型不会按格式调工具 |
| 教师 | DeepSeek-V4 | **60.0%** | 大模型直接跑 115 任务 |
| 学生 | Qwen3-4B + LoRA | **27.8%** | 蒸馏后，提升 **+20.8pp** |

## 流程

```
τ-bench 115 个任务
        │
   ┌────┴────────────┐
   │                 │
 Base 模型跑       教师模型跑
 Qwen3-4B          DeepSeek-V4
 pass^1 = 7%       pass^1 = 60%
   │                 │
   │            保留 69 条 PASS 轨迹
   │                 │
   │                 ▼
   │          转换为 ChatML 训练数据
   │                 │
   │                 ▼
   │          QLoRA 微调（rank=8, lr=1e-5）
   │                 │
   └────┬────────────┘
        ▼
   微调后 Qwen3-4B 跑 τ-bench
   pass^1 = 27.8%（+20.8pp）
```

## 关键技术

- **知识蒸馏**：大模型（DeepSeek）生成成功轨迹，小模型（Qwen3-4B）模仿学习
- **工具调用学习**：模型从轨迹中学到工具调用顺序、JSON 格式、追问策略
- **QLoRA 微调**：仅训练 28MB LoRA adapter，不改动基座权重
- **τ-bench 客观评测**：数据库状态哈希比对，非 LLM 主观打分

## 基于 DeepSeek Harness 的轨迹采集 Pipeline

除了用 τ-bench 自带 agent 采集轨迹，还基于开源 Agent 框架 [DeepSeek Harness (dsh)](https://github.com/deepseek-ai/deepseek-harness) 做了二次开发，实现了一套独立的轨迹采集 pipeline。

### 为什么用 dsh

dsh 是 DeepSeek 开源的 Agent 运行框架（TypeScript），内核是一个 `ReactLoopAgent` 状态机，负责对话循环、工具调度、会话记录。通过它的 **Python SDK**（`deepseek-harness-sdk`）可以脚本化批量驱动 Agent，而不是手动开 web UI。

### 架构

```
dsh SDK 驱动 DeepSeek 教师模型跑任务
    │
    ├─ dsh 插件 → HTTP tool bridge → τ-bench 16 个零售工具
    │   （TS 与 Python 跨语言桥接，工具在仿真环境真实执行）
    │
    ├─ dsh 自动将全过程记录为 session（append-only 事件流，zstd 压缩）
    │   事件类型：user/message, assistant/message, tool/call, tool/result
    │
    └─ session_to_chatml.py：解析事件流 → 过滤注入消息 → 折叠成 ChatML 多轮轨迹
        → 直接作为 QLoRA 训练数据
```

### 二次开发内容

| 文件 | 作用 |
|---|---|
| `collect_traces.py` | dsh Python SDK 批量跑任务，自动收集 session 文件 |
| `session_to_chatml.py` | 解析 dsh session 事件流（支持 zstd 解压），转成 ChatML |
| `tool_bridge.py` | τ-bench 工具的 HTTP 桥接，供 dsh 插件调用 |
| `dsh-plugin/` | TypeScript 插件，注册 16 个零售工具到 dsh 工具注册表 |

### 关键技术点

- **跨语言桥接**：dsh 是 TypeScript，τ-bench 是 Python，通过 HTTP 打通
- **事件流解析**：dsh 把"模型回复"和"工具调用"拆成独立事件，按 step 折叠回一条 assistant 消息
- **自动过滤注入消息**：dsh 会自动注入 runtime context（`source.kind=plugin`），训练时只保留真人消息（`source.kind=user`）
- **zstd 自动解压**：dsh session 默认 zstd 压缩，解析时自动解压

已端到端验证：dsh SDK 启动 → DeepSeek 回复 → session 落盘 → zstd 解压 → 转成干净 ChatML。

## 进阶：生产级 Agent 运行时机制

参考开源客服 Harness [Ecbot](https://github.com/blackmouse572/ecbot)，实现两个工业级运行时机制。

### 1. 生成租约（Generation Lease）：防止过时回复

模型生成回复期间，用户可能又发了新消息，回复生成完已经在回答过时问题。

机制：每条新消息递增一个计数器（epoch）；生成回复时记录当前 epoch，流式生成过程中、发送前检查 epoch 是否变化；若变化则中止生成，不发送过时回复，避免浪费 token。

```python
lease.bump(conv)              # 新消息到达
handle = lease.acquire(conv)  # 开始生成
# ... 生成中，用户又发消息 → lease.bump(conv)
handle.check_or_abort()       # 抛出 StaleGenerationError，中止
```

### 2. 智能转人工（Smart Escalation）：知道何时停下

四种触发条件：
- 客户明确要求人工（关键词 / 正则）
- 客户愤怒（情绪信号 + 历史累计）
- 检测到 prompt injection / 滥用
- Agent 置信度低于阈值

```python
decision = escalation.check(
    "I want to speak to a human! This is ridiculous!",
    agent_confidence=0.3,
)
# → should_handoff=True, reasons: customer_requested_human, angry_customer, low_confidence
```

运行演示：`python -m src.advanced.demo`

## 快速开始

```bash
# 安装依赖
pip install tau-bench litellm openai

# Base 基线评测
python eval/run_base.py 0 -1 ./results/base

# 采集教师轨迹
python eval/run_teacher.py 0 -1 ./results/teacher

# 构建训练数据
python build_dataset.py

# LoRA 微调
mlx_lm.lora --model ./models/Qwen3-4B --data ./data/lora --train --iters 100

# 微调后评测
python mlx_server.py &
python eval/run_base.py 0 -1 ./results/ft
```

## 项目结构

```
├── agent.py              # 客服 Agent 脚本
├── mlx_server.py          # OpenAI 兼容推理服务
├── build_dataset.py       # τ-bench 教师轨迹 → ChatML
├── session_to_chatml.py   # dsh session → ChatML（二次开发）
├── collect_traces.py      # dsh SDK 批量采集轨迹（二次开发）
├── tool_bridge.py         # τ-bench 工具 HTTP 桥接
├── eval/
│   ├── run_base.py       # base/微调后评测
│   └── run_teacher.py    # τ-bench 教师轨迹采集
├── src/advanced/
│   ├── generation_lease.py    # 生成租约：防过时回复
│   ├── smart_escalation.py    # 智能转人工：愤怒/注入/低置信度
│   └── demo.py                # 两个机制的演示
├── dsh-plugin/           # DeepSeek Harness 零售工具插件（TS）
├── data/lora/             # 训练数据
└── lora_adapter/          # LoRA 权重
```
