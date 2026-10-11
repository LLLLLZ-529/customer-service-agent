#!/usr/bin/env python3
"""Convert teacher trajectories to mlx-lm LoRA training data (OpenAI messages format)."""
import json
import glob
import os

results_dir = "results/teacher-full"
out_dir = "data/lora"

os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(f"{results_dir}/*.json"))
with open(files[-1]) as f:
    results = json.load(f)

all_data = []
n_traj = 0

for r in results:
    if r["reward"] != 1:
        continue
    traj = r["traj"]
    n_traj += 1

    messages = []
    for msg in traj:
        role = msg["role"]
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")

        if role == "system":
            messages.append({"role": "system", "content": content})
        elif role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            m = {"role": "assistant", "content": content or ""}
            if tool_calls:
                m["tool_calls"] = tool_calls
            messages.append(m)
        elif role == "tool":
            messages.append({"role": "tool", "content": content})

    if messages:
        all_data.append({"messages": messages})

# Split
n_train = int(len(all_data) * 0.85)
train = all_data[:n_train]
valid = all_data[n_train:]

for name, data in [("train", train), ("valid", valid)]:
    with open(f"{out_dir}/{name}.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"PASS trajectories: {n_traj}")
print(f"train: {len(train)}, valid: {len(valid)}")
print(f"Output: {out_dir}/")
