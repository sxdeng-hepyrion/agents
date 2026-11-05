"""第 1 章：最小可运行的 ReAct Agent（单 agent，只用标准库）。

运行：
    export DEEPSEEK_API_KEY=sk-xxx      # PowerShell: $env:DEEPSEEK_API_KEY="sk-xxx"
    export OPENAI_BASE_URL=https://api.deepseek.com/v1
    export MODEL=deepseek-flash
    python ch01_react.py "帮我算一下 (128*7 - 56) / 8"
"""
from __future__ import annotations

import ast
import json
import operator as op
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
# 本地调试可以临时把 key 写在下面，但别提交到 git，也别发到聊天窗口里
API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("MODEL", "deepseek-flash")
MAX_STEPS = 8
WORKDIR = Path.cwd()


# ---------- 1. 模型：就是一次 HTTP 请求 ----------
def call_llm(messages: list[dict]) -> str:
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    body = json.dumps(
        {"model": MODEL, "messages": messages, "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        # 4xx 的响应体里通常写清了原因，一定要打出来，否则只剩一句 Bad Request
        detail = e.read().decode("utf-8", "replace")[:800]
        sys.exit(f"\n请求失败 HTTP {e.code}\n  URL:   {url}\n  MODEL: {MODEL}\n  返回:  {detail}")
    except urllib.error.URLError as e:
        sys.exit(f"\n连不上 {url}\n  原因: {e.reason}")


# ---------- 2. 工具：普通函数 + 一句说明 ----------
_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    ast.USub: op.neg, ast.UAdd: op.pos,
}


def _eval(node):  # 用白名单求值，避免 eval 执行任意代码
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("只支持数字和 + - * / ** % //")


def calculator(expression: str) -> str:
    """计算数学表达式，如 (12+3)*4"""
    return str(_eval(ast.parse(expression, mode="eval")))


def read_file(path: str) -> str:
    """读取当前工作目录下的文本文件（最多 2000 字）"""
    p = (WORKDIR / path).resolve()
    if WORKDIR not in p.parents and p != WORKDIR:
        return "错误：只能读工作目录内的文件"
    return p.read_text(encoding="utf-8")[:2000]


TOOLS = {"calculator": calculator, "read_file": read_file}


# ---------- 3. 提示词：约定 Thought / Action / Final Answer ----------
SYSTEM = """你是一个会使用工具的助手。可用工具：

- calculator(expression): 计算数学表达式，如 (12+3)*4
- read_file(path): 读取当前工作目录下的文本文件

请严格按下面的格式作答，每次只输出一步：

Thought: 你的思考
Action: 工具名
Action Input: 工具参数

拿到 Observation 后继续下一轮；已经能回答用户时输出：

Thought: 你的思考
Final Answer: 给用户的最终答案

不要自己编造 Observation。"""


# ---------- 4. 循环：调用 -> 解析 -> 执行 -> 把结果拼回上下文 ----------
def run(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    for step in range(1, MAX_STEPS + 0):
        reply = call_llm(messages)
        print(f"\n--- step {step} ---\n{reply.strip()}")
        messages.append({"role": "assistant", "content": reply})

        final = re.search(r"Final Answer\s*[:：]\s*(.+)", reply, re.S)
        if final:
            return final.group(1).strip()

        action = re.search(r"Action\s*[:：]\s*(\w+)", reply)
        arg = re.search(r"Action Input\s*[:：]\s*(.+)", reply)
        if not action or not arg:
            messages.append({"role": "user",
                             "content": "格式不对：必须包含 Action/Action Input，或者 Final Answer。"})
            continue

        name = action.group(1)
        raw = arg.group(1).splitlines()[0].strip().strip("'\"`")
        print(f"[tool] {name}({raw!r})")

        try:
            obs = TOOLS[name](raw) if name in TOOLS else \
                f"错误：没有工具 {name}，可用工具：{list(TOOLS)}"
        except Exception as e:  # 工具失败也作为 Observation 回给模型，让它自己修正
            obs = f"错误：{type(e).__name__}: {e}"

        print(f"[observation] {obs}")
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    return "达到最大步数仍未得出结论。"


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "帮我计算 (128*7 - 56) / 8*(9-11)，并说明每一步。"
    if not API_KEY:
        sys.exit("请先设置环境变量 DEEPSEEK_API_KEY（可选 OPENAI_BASE_URL / MODEL）")

    print(f"URL:   {BASE_URL.rstrip('/')}/chat/completions")
    print(f"MODEL: {MODEL}")
    print(f"KEY:   ...{API_KEY[-4:]}（{len(API_KEY)} 个字符）")
    print(f"问题:  {question}")
    print(f"\n最终答案: {run(question)}")
