"""极简 MCP client：拉起一个 stdio server，握手，把它的工具变成 Tool 记录。

单独运行会做一次自检：列出工具、调一次计算器、读一个文件。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ch02_tools import Param, Tool

PROTOCOL_VERSION = "2025-06-18"


class MCPClient:
    def __init__(self, cmd: list[str], cwd: str | None = None):
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        self.proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
        self._id = 0

    # ---- 传输层：一行一条 JSON ----
    def _send(self, obj: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _read(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("server 已退出")
        return json.loads(line)

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        while True:                       # 跳过服务端发来的通知
            msg = self._read()
            if msg.get("id") == self._id:
                return msg

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # ---- 协议层：握手之后才能调工具 ----
    def initialize(self) -> dict:
        res = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "mini-agent", "version": "0.1"},
        })
        self.notify("notifications/initialized")
        return res["result"]

    def list_tools(self) -> list[dict]:
        return self.request("tools/list")["result"]["tools"]

    def call(self, name: str, args: dict) -> str:
        res = self.request("tools/call", {"name": name, "arguments": args})
        if "error" in res:
            return f"错误：{res['error']['message']}"
        blocks = res["result"].get("content", [])
        return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def load_tools(client: MCPClient) -> dict[str, Tool]:
    """把 tools/list 的结果翻译成第二章那套 Tool 记录，后面的循环不用改。"""
    tools: dict[str, Tool] = {}
    for spec in client.list_tools():
        schema = spec.get("inputSchema", {})
        required = set(schema.get("required", []))
        params = {}
        for pname, p in schema.get("properties", {}).items():
            params[pname] = Param(
                name=pname,
                type=p.get("type", "string"),
                required=pname in required,
                default=p.get("default"),
            )

        def make(name: str):
            def call(**kwargs) -> str:
                return client.call(name, kwargs)
            return call

        tools[spec["name"]] = Tool(
            name=spec["name"],
            description=spec.get("description", ""),
            params=params,
            func=make(spec["name"]),
        )
    return tools


if __name__ == "__main__":
    server = Path(__file__).with_name("mcp_server.py")
    client = MCPClient([sys.executable, str(server)], cwd=str(Path.cwd()))
    try:
        info = client.initialize()
        print("serverInfo:", info["serverInfo"], "protocol:", info["protocolVersion"])
        for t in client.list_tools():
            print("tool:", t["name"], "|", t["description"])
        print("calculator ->", client.call("calculator", {"expression": "(128*7-56)/8"}))
        print("read_file  ->", client.call("read_file", {"path": "AGENT.md", "limit": 40}))
        print("bad tool   ->", client.call("rm_rf", {}))
    finally:
        client.close()
