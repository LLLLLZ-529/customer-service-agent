#!/usr/bin/env python3
"""
Tau-bench Tool Bridge: exposes retail tools as HTTP API for dsh plugin.
dsh plugin calls this server, which executes tools in the tau-bench environment.

Usage:
    python tool_bridge.py                  # default port 9000
    python tool_bridge.py --port 9001
"""
import argparse
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add tau-bench to path
TAU_BENCH_PATH = os.path.join(os.path.dirname(__file__), "tau-bench")
sys.path.insert(0, TAU_BENCH_PATH)


class ToolBridge:
    """Manages tau-bench retail environment and tool execution."""

    def __init__(self):
        self.env = None
        self.tools = {}
        self._initialize()

    def _initialize(self):
        """Initialize retail environment and load tools."""
        try:
            from tau_bench.envs.retail.tools import ALL_TOOLS
            from tau_bench.envs.retail.retail import RetailEnv
            from tau_bench.envs.retail.retail_client import RetailClient

            # Initialize environment with default data
            client = RetailClient()
            self.env = RetailEnv(
                user_strategy="llm",
                user_model="gpt-4o",
                task_split="test",
                client=client,
            )
            self.tools = ALL_TOOLS
            print(f"Loaded {len(ALL_TOOLS)} retail tools")
        except Exception as e:
            print(f"Warning: Could not initialize full env: {e}")
            print("Bridge will run in schema-only mode")
            self.tools = {}

    def execute_tool(self, tool_name, arguments):
        """Execute a tool with given arguments."""
        if self.env is None:
            return {"error": "environment not initialized"}

        if tool_name not in self.tools:
            return {"error": f"unknown tool: {tool_name}"}

        try:
            tool = self.tools[tool_name]
            result = tool(self.env, **arguments)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def list_tools(self):
        """Return tool schemas."""
        schemas = []
        for name, tool in self.tools.items():
            schemas.append({
                "name": name,
                "description": getattr(tool, "__doc__", "") or "",
            })
        return schemas


bridge = ToolBridge()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/execute":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))

            tool_name = body.get("tool_name")
            arguments = body.get("arguments", {})

            result = bridge.execute_tool(tool_name, arguments)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result, default=str).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/tools":
            result = bridge.list_tools()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Tool bridge on http://127.0.0.1:{args.port}")
    server.serve_forever()
