#!/usr/bin/env python3
"""
E-commerce Customer Service Agent
==================================
Standalone agent that uses the fine-tuned Qwen3-4B to handle customer requests
via τ-bench's tool-calling loop.

Usage:
    python agent.py                    # interactive chat
    python agent.py --task 0            # run a specific tau-bench task
"""
import json
import os
import sys
import requests
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
MODEL_URL = "http://127.0.0.1:8000/v1/chat/completions"
SYSTEM_PROMPT = """# Retail agent policy
As a retail agent, you can help users cancel or modify pending orders, return or
exchange delivered orders, modify their default payment method, and check order
status. Always verify identity first. Use tools to look up real data — never
invent order IDs, prices, or product details.
"""


def chat(messages, max_tokens=1024, temperature=0.0):
    """Call the local MLX OpenAI-compatible server."""
    resp = requests.post(MODEL_URL, json={
        "model": "qwen3-4b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }, timeout=120)
    return resp.json()["choices"][0]["message"]["content"]


def run_interactive():
    """Simple interactive customer service chat."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("=== Customer Service Agent (type 'exit' to quit) ===\n")
    while True:
        user = input("Customer: ").strip()
        if user.lower() in ("exit", "quit"):
            break
        messages.append({"role": "user", "content": user})
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    run_interactive()
