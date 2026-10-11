#!/usr/bin/env python3
"""Simple OpenAI-compatible server using mlx_lm."""
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

MODEL_PATH = "/Users/annedl/Doubao/chats/2026-10-07/new-chat/customer-service-agent/models/Qwen3-4B-Instruct-mlx"

print("Loading model...", flush=True)
model, tokenizer = load(MODEL_PATH, adapter_path="./lora_adapter")
print("Model loaded with LoRA adapter!", flush=True)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            messages = body.get("messages", [])
            max_tokens = body.get("max_tokens", 512)
            temperature = body.get("temperature", 0.7)

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            sampler = make_sampler(temperature)
            response = ""
            for chunk in stream_generate(
                model, tokenizer, prompt=text,
                max_tokens=max_tokens, sampler=sampler
            ):
                response += chunk.text

            result = {
                "choices": [{
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop"
                }]
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            result = {"data": [{"id": "qwen3-4b", "object": "model"}]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8000), Handler)
    print("Server on http://127.0.0.1:8000", flush=True)
    server.serve_forever()
