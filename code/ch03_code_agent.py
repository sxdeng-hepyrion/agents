"""第 3 章附例：一个 code agent 的最小工具集。

五个工具：list_dir / view / search / edit / run_command。
前四个是结构化的只读与改写操作，最后一个是 shell，能力最强也最危险。

运行：
    export DEEPSEEK_API_KEY=sk-xxx
    python ch03_code_agent.py "看看工作目录里有什么，然后告诉我 chapters 下有几个文件"
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from ch02_tools import TOOLS, ask_user, call_tool, tool
from ch03_function_calling import run

WORKDIR = Path.cwd()
SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea"}


def _safe(path: str) -> Path:
    p = (WORKDIR / path).resolve()
    if WORKDIR not in p.parents and p != WORKDIR:
        raise ValueError(f"路径越界，只能访问工作目录：{path}")
    return p


@tool("列出目录下的文件和子目录，不递归，目录以 / 结尾")
def list_dir(path: str = ".") -> str:
    p = _safe(path)
    if not p.is_dir():
        return f"不是目录：{path}"
    items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    lines = [f"{e.name}/" if e.is_dir() else e.name for e in items if e.name not in SKIP]
    return "\n".join(lines[:200]) or "（空目录）"


@tool("按行号查看文件内容，start 和 end 是 1 起始的行号，适合大文件分段读")
def view(path: str, start: int = 1, end: int = 200) -> str:
    p = _safe(path)
    if not p.exists():
        return f"文件不存在：{path}"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, start)
    chunk = lines[start - 1:end]
    body = "\n".join(f"{i:>5}| {line}" for i, line in enumerate(chunk, start))
    if end < len(lines):
        body += f"\n（共 {len(lines)} 行，还有 {len(lines) - end} 行未显示）"
    return body or "（空文件）"


@tool("在工作目录内按正则搜索文件内容，返回 文件:行号:内容")
def search(pattern: str, path: str = ".", max_results: int = 40) -> str:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"正则不合法：{e}"
    hits: list[str] = []
    root = _safe(path)
    for f in sorted(root.rglob("*")):
        if not f.is_file() or SKIP & set(f.parts):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for no, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                rel = f.relative_to(WORKDIR).as_posix()
                hits.append(f"{rel}:{no}: {line.strip()[:160]}")
                if len(hits) >= max_results:
                    return "\n".join(hits) + f"\n（已到 {max_results} 条上限，可能还有更多）"
    return "\n".join(hits) or "没有匹配"


@tool("把文件中一段文本替换成新的文本，old_text 必须恰好出现一次", requires_confirm=True)
def edit(path: str, old_text: str, new_text: str) -> str:
    p = _safe(path)
    if not p.exists():
        return f"文件不存在：{path}"
    text = p.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count == 0:
        return (f"没有找到要替换的内容：{path}，共 {len(text.splitlines())} 行。"
                f"请先用 view 确认原文，注意缩进和空行要完全一致")
    if count > 1:
        return f"这段文本在文件里出现了 {count} 次，请给出更长的一段以便唯一定位"
    p.write_text(text.replace(old_text, new_text), encoding="utf-8")
    return f"已修改 {p.name}：替换了 {len(old_text)} 个字符，写入 {len(new_text)} 个字符"


@tool("在工作目录执行一条 shell 命令，返回退出码和输出；每次都是新进程，cd 和 export 不会保留",
      requires_confirm=True)
def run_command(command: str, timeout: int = 30) -> str:
    try:
        # 输出先落盘再读，避免大输出把管道缓冲区撑住
        with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as f:
            r = subprocess.run(
                command, shell=True, cwd=WORKDIR,
                stdout=f, stderr=subprocess.STDOUT,
                text=True, timeout=timeout, stdin=subprocess.DEVNULL,
            )
            f.seek(0)
            out = f.read(8000).strip()
    except subprocess.TimeoutExpired:
        return f"超时：命令超过 {timeout} 秒，已被终止"
    if len(out) > 4000:
        out = out[:4000] + "\n（输出已截断）"
    return f"exit={r.returncode}\n{out}" if out else f"exit={r.returncode}（无输出）"


CODE_TOOLS = {name: TOOLS[name] for name in
              ("list_dir", "view", "search", "edit", "run_command")}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "看看工作目录里有什么，然后告诉我 chapters 下有几个文件"
    if not os.getenv("DEEPSEEK_API_KEY"):
        sys.exit("请先设置环境变量 DEEPSEEK_API_KEY")
    print(f"工具:  {', '.join(CODE_TOOLS)}")
    print(f"问题:  {question}")
    print(f"\n最终答案: {run(question, CODE_TOOLS, call_tool, ask_user)}")
