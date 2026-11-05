"""第 3 章附例：把工具搬到 MCP 后面，循环一行都不用改。

启动 mcp_server.py 作为子进程，握手、拿工具、再交给 ch03 的 run()。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ch03_function_calling import run
from mcp_client import MCPClient, load_tools


def mcp_caller(client: MCPClient):
    def call(name: str, raw: str, approve=None) -> str:
        try:
            args, _ = json.JSONDecoder().raw_decode(raw.strip().strip("`"))
        except json.JSONDecodeError:
            return "错误：参数必须是 JSON 对象"
        if not isinstance(args, dict):
            return "错误：参数必须是 JSON 对象"
        return client.call(name, args)
    return call


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "算一下 (128*7-56)/8，再读 AGENT.md 的第一行核对一下项目名"

    server = Path(__file__).with_name("mcp_server.py")
    client = MCPClient([sys.executable, str(server)])
    try:
        info = client.initialize()
        print(f"MCP server: {info['serverInfo']['name']} {info['serverInfo']['version']} "
              f"(protocol {info['protocolVersion']})")

        tools = load_tools(client)
        print(f"工具:  {', '.join(tools)}")
        print(f"问题:  {question}")
        print(f"\n最终答案: {run(question, tools, mcp_caller(client))}")
    finally:
        client.close()
