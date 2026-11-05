"""第 3 章：用原生 function calling 替掉文本协议。

工具来自第二章的注册表（ch02_tools.py）。这一章只改"模型怎么表达调用"，
并且把 run() 写成可以接受任意工具集的形状，后面接 MCP 时可以直接复用。

运行：
    export DEEPSEEK_API_KEY=sk-xxx      # PowerShell: $env:DEEPSEEK_API_KEY="sk-xxx"
    python ch03_function_calling.py "读一下 AGENT.md 的第一节，用一句话概括"
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Callable

from ch02_tools import TOOLS, Tool, ask_user, call_tool

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("MODEL", "deepseek-flash")
MAX_STEPS = 8

SYSTEM = "你是一个会使用工具的助手。需要计算、读文件或写文件时，调用相应的工具；拿到结果后再回答用户。"

Caller = Callable[[str, str, Callable[[str, dict], bool] | None], str]


# ---------- 1. 把工具记录翻译成 JSON Schema ----------
def tool_schema(t: Tool) -> dict:
    props: dict[str, dict] = {}
    required: list[str] = []
    for p in t.params.values():
        props[p.name] = {"type": p.type}
        if p.required:
            required.append(p.name)
        else:
            props[p.name]["default"] = p.default
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


def build_schema(tools: dict[str, Tool]) -> list[dict]:
    return [tool_schema(t) for t in tools.values()]


# ---------- 2. 一次请求 ----------
def call_llm(messages: list[dict], schema: list[dict]) -> dict:
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    body: dict = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
    }
    if schema:  # 没有工具时不要发 "tools": []，有的服务端会因此报错
        body["tools"] = schema
        body["tool_choice"] = "auto"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["choices"][0]["message"]
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:800]
        sys.exit(f"\n请求失败 HTTP {e.code}\n  URL:   {url}\n  MODEL: {MODEL}\n  返回:  {detail}")
    except urllib.error.URLError as e:
        sys.exit(f"\n连不上 {url}\n  原因: {e.reason}")


# ---------- 3. 循环：看 tool_calls，结果以 tool 角色回填 ----------
def run(question: str, tools: dict[str, Tool], caller: Caller, approve=ask_user) -> str:
    schema = build_schema(tools)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    for step in range(1, MAX_STEPS + 1):
        msg = call_llm(messages, schema)
        calls = msg.get("tool_calls") or []
        print(f"\n--- step {step} ---")
        if msg.get("content"):
            print(msg["content"].strip())

        # assistant 这条原样存回去，tool_calls 的 id 后面要用来对应结果
        messages.append(msg)

        if not calls:
            return (msg.get("content") or "").strip()

        for c in calls:
            fn = c["function"]
            print(f"[tool] {fn['name']}({fn['arguments']})")
            obs = caller(fn["name"], fn["arguments"], approve)
            print(f"[observation] {obs}")
            messages.append({
                "role": "tool",
                "tool_call_id": c["id"],
                "content": obs,
            })

    return "达到最大步数仍未得出结论。"


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "读一下 AGENT.md 的第一节，用一句话概括"
    if not API_KEY:
        sys.exit("请先设置环境变量 DEEPSEEK_API_KEY（可选 OPENAI_BASE_URL / MODEL）")

    print(f"URL:   {BASE_URL.rstrip('/')}/chat/completions")
    print(f"MODEL: {MODEL}")
    print(f"工具:  {', '.join(TOOLS)}")
    print(f"问题:  {question}")
    print(f"\n最终答案: {run(question, TOOLS, call_tool)}")
