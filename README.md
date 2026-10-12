# 电商客服 Agent：基于教师轨迹蒸馏的工具调用能力微调

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
├── build_dataset.py       # 教师轨迹 → ChatML 训练数据
├── tool_bridge.py         # τ-bench 工具 HTTP 桥接
├── eval/
│   ├── run_base.py       # base/微调后评测
│   └── run_teacher.py    # 教师轨迹采集
├── src/advanced/
│   ├── generation_lease.py    # 生成租约：防过时回复
│   ├── smart_escalation.py    # 智能转人工：愤怒/注入/低置信度
│   └── demo.py                # 两个机制的演示
├── dsh-plugin/           # DeepSeek Harness 零售工具插件
├── data/lora/             # 训练数据
└── lora_adapter/          # LoRA 权重
```
