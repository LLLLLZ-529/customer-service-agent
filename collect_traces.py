#!/usr/bin/env python3
"""
collect_traces.py
=================
用 DeepSeek Harness 的 Python SDK 批量驱动 τ-bench 任务，
自动采集教师模型（DeepSeek）的成功轨迹，导出成 ChatML 训练数据。

整条链路：
  启动 tool_bridge（τ-bench 工具 HTTP 服务）
    → 启动 dsh harness（Python SDK，挂载零售工具插件）
    → 循环跑 τ-bench 115 个任务
    → 每个任务结束后拿 reward（PASS/FAIL）
    → 对 PASS 的任务，定位 dsh session.jsonl
    → 调 session_to_chatml 转成 ChatML，追加进 train.jsonl

依赖：
  pip install deepseek-harness-sdk
  export DEEPSEEK_API_KEY=sk-xxx

用法：
  python3 collect_traces.py --limit 5        # 先跑 5 个任务试通
  python3 collect_traces.py --limit 115      # 跑全部
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# 复用已写好的转换脚本
sys.path.insert(0, str(Path(__file__).parent))
from session_to_chatml import load_session_events, events_to_messages


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
DSH_HOME = Path("./dsh-home").resolve()        # dsh 运行时数据目录（session 存在这里）
TOOL_BRIDGE_PORT = 9000
OUTPUT_DIR = Path("./data/traces").resolve()   # 导出的 ChatML 存这里


def start_tool_bridge():
    """启动 τ-bench 工具的 HTTP 桥接（tool_bridge.py）。

    tool_bridge 把 τ-bench 的 16 个工具暴露成 POST /execute，
    dsh 插件通过 HTTP 调用它们。
    """
    print("[setup] 启动 tool_bridge (τ-bench 工具 HTTP 服务)...")
    proc = subprocess.Popen(
        [sys.executable, "tool_bridge.py"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    # 等它监听 9000 端口
    time.sleep(3)
    print("[setup] tool_bridge 已启动 (端口 9000)")
    return proc


def find_session_jsonl(session_id: str) -> Path | None:
    """根据 session_id，在 DSH_HOME 下找到对应的 .jsonl 轨迹文件。

    dsh 把 session 持久化成 JSONL，文件名里通常含 session id。
    实际路径需要在首次跑通后确认，这里给一个通用的查找逻辑。
    """
    candidates = list(DSH_HOME.rglob(f"*{session_id}*.jsonl"))
    if candidates:
        return candidates[0]
    # 兜底：找最新的 jsonl
    all_jsonl = sorted(DSH_HOME.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return all_jsonl[-1] if all_jsonl else None


def run_one_task(harness, task_id: str, initial_user_msg: str) -> dict:
    """跑单个 τ-bench 任务。

    参数：
      harness: dsh 的 DeepSeekHarness 实例
      task_id: τ-bench 任务编号（如 "001"）
      initial_user_msg: τ-bench 给的第一条用户消息

    返回：
      { task_id, reward, session_path, final_response }
    """
    print(f"\n[task {task_id}] 开始")
    t0 = time.time()

    # 核心：让 dsh 跑这个任务。
    # harness.run 会驱动完整 Agent loop：
    #   用户消息 → 模型思考 → 调工具（经 tool_bridge 操作 τ-bench）→ 结果回传 → 继续 → 完成
    result = harness.run(
        initial_user_msg,
        session_id=f"task-{task_id}",
    )

    elapsed = time.time() - t0
    print(f"[task {task_id}] 完成，耗时 {elapsed:.1f}s")

    # 找这次跑的 session 文件
    session_path = find_session_jsonl(f"task-{task_id}")

    return {
        "task_id": task_id,
        "session_path": str(session_path) if session_path else None,
        "final_response": result.final_response[:200],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5, help="跑几个任务（先小批量试通）")
    ap.add_argument("--model", default="deepseek-v4-flash", help="教师模型")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DSH_HOME.mkdir(parents=True, exist_ok=True)

    # 1. 启动 tool_bridge
    bridge = start_tool_bridge()

    # 2. 启动 dsh harness（Python SDK）
    #    参考：dsh python/sdk 文档
    from deepseek_harness import DeepSeekHarness

    print("[setup] 启动 dsh harness...")
    harness = DeepSeekHarness(
        dsh_home=str(DSH_HOME),
        cwd=str(Path.cwd()),
        provider="deepseek-official",
        model=args.model,
        # patches=("dsh-plugin/cordis.patch.yml",),  # 挂载我们的零售工具插件
    )

    # 3. 准备 τ-bench 任务列表
    #    这里替换成你实际从 τ-bench 加载任务的方式：
    #    from tau_bench.envs.tool import Tool
    #    env = Tool()
    #    tasks = list(...)  # 115 个任务的初始消息
    tasks = [
        # 示例格式：(task_id, initial_user_message)
        ("001", "Hi, I want to check my order status. My email is test@test.com"),
        ("002", "I need to return a delivered order. My name is Alice, zip 12345"),
    ][: args.limit]

    # 4. 批量跑 + 收集
    results = []
    for task_id, user_msg in tasks:
        r = run_one_task(harness, task_id, user_msg)
        results.append(r)

    # 5. 导出成功轨迹（这里 reward 需要从 τ-bench 环境读，先占位）
    print(f"\n[export] 转换 {len(results)} 个 session → ChatML")
    out_file = OUTPUT_DIR / "train.jsonl"
    with open(out_file, "w", encoding="utf-8") as fout:
        for r in results:
            if not r["session_path"]:
                continue
            # reward 判断：从 τ-bench env 读（这里简化为全部导出，实际按 reward 过滤）
            events = load_session_events(Path(r["session_path"]))
            messages = events_to_messages(events)
            if any(m["role"] == "user" for m in messages):
                fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"\n[done] 导出完成 → {out_file}")
    print("       下一步：把 train.jsonl 喂给 QLoRA 训练")

    # 清理
    harness.close()
    bridge.terminate()


if __name__ == "__main__":
    main()
