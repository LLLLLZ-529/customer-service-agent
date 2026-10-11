#!/usr/bin/env python3
"""Run tau-bench with local MLX model as agent."""
import os
import sys

# Load .env
from pathlib import Path
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Point OpenAI-compatible client to local MLX server
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:8000/v1"
os.environ["OPENAI_API_KEY"] = "dummy"

from tau_bench.run import run
from tau_bench.types import RunConfig

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end = int(sys.argv[2]) if len(sys.argv) > 2 else 3
out_dir = sys.argv[3] if len(sys.argv) > 3 else "./results/base-smoke"

config = RunConfig(
    model_provider="openai",
    user_model_provider="deepseek",
    model="qwen3-4b",
    user_model="deepseek-chat",
    env="retail",
    agent_strategy="tool-calling",
    temperature=0.0,
    task_split="test",
    start_index=start,
    end_index=end,
    log_dir=out_dir,
    max_concurrency=1,
    seed=10,
    user_strategy="llm",
)

results = run(config)
n_pass = sum(1 for r in results if r.reward == 1)
print(f"\n=== {n_pass}/{len(results)} PASS ===")
