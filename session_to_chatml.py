#!/usr/bin/env python3
"""
session_to_chatml.py
====================
把 DeepSeek Harness (dsh) 的 session 轨迹日志（append-only JSONL）
转换成 QLoRA 训练用的 ChatML 格式。

dsh session 真实格式（v4）——每行一个 JSON 事件：
  {"type": "user/message",     "data": {"content": [{"type":"text","text":"..."}]}}
  {"type": "assistant/message", "data": {"turn":1,"step":1,"message":{"content":[...]}}}
  {"type": "tool/call",        "data": {"turn":1,"step":1,"callId":"...","name":"...","arguments":"{...json...}"}}
  {"type": "tool/result",      "data": {"turn":1,"step":1,"message":{"toolCallId":"...","content":[{"type":"text","text":"..."}]}}}

本脚本做的事：
  1. 逐行读 jsonl，只挑出上面四类事件
  2. 把 assistant/message 和同 step 的 tool/call 聚合成一条 assistant 消息（含 tool_calls）
  3. 把 tool/result 转成 role="tool" 的消息
  4. 输出 {"messages": [...]}，与现有 build_dataset.py 格式一致

用法：
  python3 session_to_chatml.py session.jsonl --reward 1 -o train.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def extract_text(content_blocks: list) -> str:
    """从 content blocks 里提取纯文本。

    dsh 的 content 是数组，每个 block 有 type：
      - "text"      → 模型要说的话（保留）
      - "reasoning" → 模型的思考链（训练时丢弃，不进 messages）
    """
    parts = []
    for block in content_blocks:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts).strip()


def load_session_events(path: Path) -> list:
    """读 jsonl，逐行解析成事件列表。"""
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 跳过损坏行
    return events


def events_to_messages(events: list) -> list:
    """把事件流折叠成 ChatML messages 数组。

    核心难点：dsh 把"模型回复"和"它要调的工具"拆成了独立事件。
    我们要按 (turn, step) 把它们重新拼回一条 assistant 消息。
    """
    messages = []
    # 暂存当前 assistant 状态：{text, tool_calls}
    pending_assistant = None

    def flush_assistant():
        """把攒好的 assistant 消息写入 messages。"""
        nonlocal pending_assistant
        if pending_assistant is None:
            return
        msg = {"role": "assistant"}
        # 只有文本时放 content；有 tool_calls 时也要带 content（可空串）
        msg["content"] = pending_assistant["text"] or ""
        if pending_assistant["tool_calls"]:
            msg["tool_calls"] = pending_assistant["tool_calls"]
        messages.append(msg)
        pending_assistant = None

    for ev in events:
        etype = ev.get("type")
        data = ev.get("data", {})

        # ① 用户消息
        if etype == "user/message":
            flush_assistant()  # 上一轮的 assistant 先落地
            text = extract_text(data.get("content", []))
            if text:
                messages.append({"role": "user", "content": text})

        # ② 模型回复（可能含 reasoning，可能要调工具）
        elif etype == "assistant/message":
            msg = data.get("message", {})
            text = extract_text(msg.get("content", []))
            # 新一轮 assistant，开启暂存
            pending_assistant = {"text": text, "tool_calls": []}

        # ③ 工具调用（独立事件，要并入上一条 assistant）
        elif etype == "tool/call":
            if pending_assistant is None:
                pending_assistant = {"text": "", "tool_calls": []}
            # arguments 是 JSON 字符串，解析成 dict 再序列化
            raw_args = data.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            pending_assistant["tool_calls"].append({
                "id": data.get("callId"),
                "type": "function",
                "function": {
                    "name": data.get("name"),
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })

        # ④ 工具结果
        elif etype == "tool/result":
            flush_assistant()  # 工具结果前，先把 assistant（含 tool_calls）落地
            msg = data.get("message", {})
            text = extract_text(msg.get("content", []))
            messages.append({
                "role": "tool",
                "tool_call_id": msg.get("toolCallId"),
                "content": text,
            })

    # 文件结束，把最后一条 assistant 落地
    flush_assistant()
    return messages


def main():
    ap = argparse.ArgumentParser(description="dsh session → ChatML")
    ap.add_argument("session", type=Path, help="dsh session jsonl 文件路径")
    ap.add_argument("--reward", type=int, default=1,
                    help="该任务是否成功（1=PASS，0=FAIL）。默认只导出成功轨迹")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="输出 jsonl 路径；不指定则打印到 stdout")
    args = ap.parse_args()

    events = load_session_events(args.session)
    messages = events_to_messages(events)

    # 过滤：至少要有 user + assistant 才算有效轨迹
    if not any(m["role"] == "user" for m in messages):
        print(f"[skip] {args.session.name}: 没有 user 消息", file=sys.stderr)
        sys.exit(0)

    sample = {"messages": messages}

    if args.output:
        # 追加写（批量导出时一次写多条）
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        n_turns = sum(1 for m in messages if m["role"] == "user")
        print(f"[ok] {args.session.name}: {len(messages)} 条消息 / {n_turns} 轮对话 → {args.output}")
    else:
        print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
