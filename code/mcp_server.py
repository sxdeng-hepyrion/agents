"""极简 MCP server：stdio 传输 + JSON-RPC 2.0。

协议消息从 stdin 逐行读入，响应从 stdout 逐行写出。
stdout 里只能出现协议消息，日志一律走 stderr，否则客户端会解析失败。

单独运行没有意义，它等着客户端（mcp_client.py）来握手。
"""
from __future__ import annotations

import json
import sys

from ch02_tools import calculator, read_file

PROTOCOL_VERSION = "2025-06-18"

# 服务端暴露的能力。这里只给只读工具：写文件那种有副作用的操作，
# 在真实场景里要么不给，要么在服务端自己做权限判断。
TOOLS = {
    "calculator": {
        "description": "计算数学表达式，例如 (12+3)*4",
        "inputSchema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        "func": calculator,
    },
    "read_file": {
        "description": "读取工作目录下的文本文件，返回前 limit 个字符",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "default": 2000},
            },
            "required": ["path"],
        },
        "func": read_file,
    },
}


def ok(mid, result):
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def err(mid, code, message):
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def handle(msg: dict) -> dict | None:
    method = msg.get("method")
    mid = msg.get("id")

    if method == "initialize":
        return ok(mid, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mini-files", "version": "0.1"},
        })

    if method == "tools/list":
        return ok(mid, {"tools": [
            {"name": name, "description": t["description"], "inputSchema": t["inputSchema"]}
            for name, t in TOOLS.items()
        ]})

    if method == "tools/call":
        params = msg.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        if name not in TOOLS:
            return err(mid, -32602, f"没有工具 {name}")
        try:
            text = str(TOOLS[name]["func"](**args))
            return ok(mid, {"content": [{"type": "text", "text": text}], "isError": False})
        except Exception as e:
            # 工具内部的失败不算协议错误：返回 isError，让客户端把话带给模型
            return ok(mid, {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}],
                            "isError": True})

    if method == "ping":
        return ok(mid, {})

    if mid is None:  # 通知，不需要回复
        return None

    return err(mid, -32601, f"不支持的方法 {method}")


def main() -> None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print("无法解析的消息", file=sys.stderr)
            continue
        resp = handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
