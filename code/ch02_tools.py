"""第 2 章：把工具变成一份带 schema 的注册表。

运行：
    export DEEPSEEK_API_KEY=sk-xxx      # PowerShell: $env:DEEPSEEK_API_KEY="sk-xxx"
    python ch02_tools.py "帮我算 (128*7-56)/8，把结果写进 result.txt"
"""
from __future__ import annotations

import ast
import inspect
import json
import operator as op
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, get_type_hints

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("MODEL", "deepseek-flash")
MAX_STEPS = 8
WORKDIR = Path.cwd()


# ---------- 1. 工具是一条记录：名字、说明、参数表、要不要确认 ----------
@dataclass
class Param:
    name: str
    type: str            # string / integer / number / boolean
    required: bool = True
    default: Any = None


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, Param]
    func: Callable[..., str]
    requires_confirm: bool = False


TOOLS: dict[str, Tool] = {}
_TYPE_NAMES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def tool(desc: str = "", requires_confirm: bool = False):
    """把函数登记成工具，参数表从签名和类型注解里读。"""
    def wrap(fn: Callable[..., str]) -> Callable[..., str]:
        hints = get_type_hints(fn)
        params: dict[str, Param] = {}
        for pname, p in inspect.signature(fn).parameters.items():
            py_type = hints.get(pname, str)
            has_default = p.default is not inspect.Parameter.empty
            params[pname] = Param(
                name=pname,
                type=_TYPE_NAMES.get(py_type, "string"),
                required=not has_default,
                default=None if not has_default else p.default,
            )
        TOOLS[fn.__name__] = Tool(
            name=fn.__name__,
            description=desc or (fn.__doc__ or "").strip(),
            params=params,
            func=fn,
            requires_confirm=requires_confirm,
        )
        return fn
    return wrap


def signature(t: Tool) -> str:
    parts = []
    for p in t.params.values():
        s = f"{p.name}: {p.type}"
        if not p.required:
            s += f" = {p.default!r}"
        parts.append(s)
    return ", ".join(parts)


# ---------- 2. 具体工具：函数怎么写，工具清单就怎么长 ----------
_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    ast.USub: op.neg, ast.UAdd: op.pos,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("只支持数字和 + - * / ** % //")


def _safe_path(path: str) -> Path:
    p = (WORKDIR / path).resolve()
    if WORKDIR not in p.parents and p != WORKDIR:
        raise ValueError(f"路径越界，只能访问工作目录：{path}")
    return p


@tool("计算数学表达式，例如 (12+3)*4")
def calculator(expression: str) -> str:
    return str(_eval(ast.parse(expression, mode="eval")))


@tool("读取工作目录下的文本文件，返回前 limit 个字符")
def read_file(path: str, limit: int = 2000) -> str:
    p = _safe_path(path)
    if not p.exists():
        return f"文件不存在：{path}"
    return p.read_text(encoding="utf-8")[:limit]


@tool("把文本写入工作目录下的文件，会覆盖原内容", requires_confirm=True)
def write_file(path: str, content: str) -> str:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {p.name}，共 {len(content)} 个字符"


# ---------- 3. 调用：解析 JSON → 校验参数 → 确认 → 执行 ----------
_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


def _type_ok(value: Any, tname: str) -> bool:
    if tname == "boolean":
        return isinstance(value, bool)
    if tname in ("integer", "number"):
        # bool 是 int 的子类，要单独排除
        return not isinstance(value, bool) and isinstance(value, _TYPES[tname])
    return isinstance(value, str)


def _example(t: Tool) -> str:
    return json.dumps({p.name: f"<{p.type}>" for p in t.params.values()}, ensure_ascii=False)


def call_tool(name: str, raw: str, approve: Callable[[str, dict], bool] | None = None) -> str:
    t = TOOLS.get(name)
    if t is None:
        return f"错误：没有工具 {name}。可用工具：{', '.join(TOOLS)}"

    try:
        args, _ = json.JSONDecoder().raw_decode(raw.strip().strip("`"))
    except json.JSONDecodeError:
        return f"错误：Action Input 必须是 JSON 对象，例如 {_example(t)}"
    if not isinstance(args, dict):
        return f"错误：Action Input 必须是 JSON 对象，例如 {_example(t)}"

    problems = []
    for p in t.params.values():
        if p.name not in args:
            if p.required:
                problems.append(f"缺少必填参数 {p.name}")
            continue
        if not _type_ok(args[p.name], p.type):
            problems.append(f"参数 {p.name} 应该是 {p.type}，收到的是 {type(args[p.name]).__name__}")
    for key in args:
        if key not in t.params:
            problems.append(f"未知参数 {key}")
    if problems:
        return "错误：" + "；".join(problems) + f"。本工具的签名是 {t.name}({signature(t)})"

    if t.requires_confirm and approve is not None and not approve(t.name, args):
        return "用户拒绝了这次调用，什么也没有执行。"

    try:
        return str(t.func(**args))
    except ValueError as e:  # 路径越界这类校验错误，直接把话说给模型听
        return f"错误：{e}"
    except Exception as e:
        return f"错误：{type(e).__name__}: {e}"


# ---------- 4. 提示词：从注册表生成，不再手抄 ----------
def system_prompt() -> str:
    lines = []
    for t in TOOLS.values():
        lines.append(f"- {t.name}({signature(t)})：{t.description}")
    tools = "\n".join(lines)
    return f"""你是一个会使用工具的助手。可用工具：

{tools}

Action Input 必须是一个 JSON 对象，键名与参数名一致，例如：
Action: read_file
Action Input: {{"path": "notes.txt", "limit": 500}}

每次只输出一步：

Thought: 你的思考
Action: 工具名
Action Input: JSON 参数

可以回答用户时输出：

Thought: 你的思考
Final Answer: 给用户的最终答案

不要自己编造 Observation。"""


# ---------- 5. 模型调用与主循环 ----------
def call_llm(messages: list[dict]) -> str:
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": 0}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:800]
        sys.exit(f"\n请求失败 HTTP {e.code}\n  URL:   {url}\n  MODEL: {MODEL}\n  返回:  {detail}")
    except urllib.error.URLError as e:
        sys.exit(f"\n连不上 {url}\n  原因: {e.reason}")


def ask_user(name: str, args: dict) -> bool:
    print(f"\n[需要确认] {name}({json.dumps(args, ensure_ascii=False)})")
    try:
        return input("执行吗？[y/N] ").strip().lower() == "y"
    except EOFError:
        return False


def run(question: str, approve=ask_user) -> str:
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": question},
    ]

    for step in range(1, MAX_STEPS + 1):
        reply = call_llm(messages)
        print(f"\n--- step {step} ---\n{reply.strip()}")
        messages.append({"role": "assistant", "content": reply})

        final = re.search(r"Final Answer\s*[:：]\s*(.+)", reply, re.S)
        if final:
            return final.group(1).strip()

        action = re.search(r"Action\s*[:：]\s*(\w+)", reply)
        arg = re.search(r"Action Input\s*[:：]\s*(.*)", reply, re.S)
        if not action or not arg:
            messages.append({"role": "user",
                             "content": "格式不对：要有 Action 和 Action Input（JSON），或者 Final Answer。"})
            continue

        name = action.group(1)
        print(f"[tool] {name}({arg.group(1).strip()})")
        obs = call_tool(name, arg.group(1), approve)
        print(f"[observation] {obs}")
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    return "达到最大步数仍未得出结论。"


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "帮我算 (128*7-56)/8，把结果写进 result.txt"
    if not API_KEY:
        sys.exit("请先设置环境变量 DEEPSEEK_API_KEY（可选 OPENAI_BASE_URL / MODEL）")

    print(f"URL:   {BASE_URL.rstrip('/')}/chat/completions")
    print(f"MODEL: {MODEL}")
    print(f"问题:  {question}")
    print(f"\n最终答案: {run(question)}")
