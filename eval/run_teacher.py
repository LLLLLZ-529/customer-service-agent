#!/usr/bin/env python3
"""Collect teacher trajectories using DeepSeek as agent."""
import os
import sys
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Point OpenAI-compatible client to DashScope (qwen-plus as user simulator)
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["OPENAI_API_KEY"] = os.environ.get("DASHSCOPE_API_KEY", "")

from tau_bench.run import run
from tau_bench.types import RunConfig

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end = int(sys.argv[2]) if len(sys.argv) > 2 else -1
out_dir = sys.argv[3] if len(sys.argv) > 3 else "./results/teacher"

config = RunConfig(
    model_provider="deepseek",
    user_model_provider="openai",
    model="deepseek-chat",
    user_model="qwen-plus",
    env="retail",
    agent_strategy="tool-calling",
    temperature=0.0,
    task_split="test",
    start_index=start,
    end_index=end,
    log_dir=out_dir,
    max_concurrency=4,
    seed=10,
    user_strategy="llm",
)

results = run(config)
n_pass = sum(1 for r in results if r.reward == 1)
print(f"\n=== Teacher: {n_pass}/{len(results)} PASS ===")
