# E-commerce Customer Service Agent — Distillation Pipeline

End-to-end pipeline: **collect teacher trajectories → SFT small model → evaluate with τ-bench**.

## Results

| Model | pass^1 (τ-bench retail) |
|---|---|
| Qwen3-4B-Instruct (base) | 7.0% |
| DeepSeek-V4 (teacher) | 60.0% |
| Qwen3-4B + QLoRA (student) | 27.8% |

## Pipeline

```
τ-bench 115 test tasks
        │
   ┌────┴─────┐
   │          │
 Base eval   Teacher trajectory collection
 pass^1=7%   69/115 PASS → save trajectories
   │          │
   │          ▼
   │    build_dataset.py (filter PASS, format ChatML)
   │          │
   │          ▼
   │    data/lora/train.jsonl (58 dialogs)
   │          │
   │          ▼
   │    mlx_lm.lora (QLoRA, rank=8, lr=1e-5)
   │          │
   │          ▼
   │    lora_adapter/ (28MB)
   │          │
   └────┬─────┘
        ▼
   Fine-tuned eval
   pass^1
```

## Quick Start

### 1. Setup

```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install tau-bench litellm openai requests

# Download Qwen3-4B-Instruct (8GB)
# from ModelScope: Qwen/Qwen3-4B-Instruct-2507
```

### 2. Run base evaluation

```bash
# Set API keys in .env
cp .env.example .env

# Run base model (local MLX server) on 115 tasks
python eval/run_base.py 0 -1 ./results/base-full
```

### 3. Collect teacher trajectories

```bash
# DeepSeek as teacher, qwen-plus as user simulator
python eval/run_teacher.py 0 -1 ./results/teacher-full
```

### 4. Build training dataset

```bash
# Filter PASS trajectories → ChatML format
python build_dataset.py
```

### 5. LoRA fine-tuning (Mac / Apple Silicon)

```bash
mlx_lm.lora \
  --model ./models/Qwen3-4B-Instruct-mlx \
  --data ./data/lora \
  --train \
  --iters 100 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --max-seq-length 2048 \
  --adapter-path ./lora_adapter
```

### 6. Evaluate fine-tuned model

```bash
# Start local server with LoRA adapter
python mlx_server.py

# Run evaluation
python eval/run_base.py 0 -1 ./results/ft-full
```

## Project Structure

```
├── agent.py              # Standalone customer service agent
├── mlx_server.py         # OpenAI-compatible MLX inference server
├── build_dataset.py      # Teacher trajectories → ChatML training data
├── eval/
│   ├── run_base.py       # Evaluate base / fine-tuned model
│   └── run_teacher.py    # Collect teacher (DeepSeek) trajectories
├── data/
│   └── lora/
│       ├── train.jsonl   # 58 training dialogs
│       └── valid.jsonl   # 11 validation dialogs
├── lora_adapter/         # Trained LoRA weights (28MB)
└── results/              # Evaluation results (JSON)
```

## Key Tech

- **τ-bench**: Retail customer service simulation (16 tools, 115 test tasks, DB-state hash scoring)
- **MLX**: Apple Silicon local inference + QLoRA training
- **Knowledge Distillation**: DeepSeek (teacher) → Qwen3-4B (student)
- **ChatML**: Tool-call preserving conversation format
