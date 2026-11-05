"""第 4 章：记忆。滑动窗口 + 摘要压缩 + 跨会话检索。

单轮：
    python ch04_memory.py "我叫小明，正在做一个 agent 学习项目"
    python ch04_memory.py "我的项目是什么？"      # 另起一个进程，靠 memory.jsonl 召回
多轮：
    python ch04_memory.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ch02_tools import TOOLS, call_tool
from ch03_function_calling import build_schema, call_llm

SCHEMA = build_schema(TOOLS)
BUDGET = 2600                 # 上下文预算，按字符数近似，不是真实 token
KEEP_RECENT = 6               # 预算超了也至少保留最近这么多条原文
MAX_STEPS = 6
MEM_FILE = Path("memory.jsonl")

SYSTEM = "你是一个会使用工具的助手。需要计算、读文件时调用工具；回答尽量简短。"


# ---------- 1. 长期记忆：一个只会追加的 jsonl + 粗略检索 ----------
def _grams(s: str) -> set[str]:
    """按字符二元组切词：对中文够用，也不需要分词依赖。"""
    s = re.sub(r"\s+", "", s.lower())
    return {s[i:i + 2] for i in range(max(0, len(s) - 1))}


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.entries: list[dict] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.entries.append(json.loads(line))

    def add(self, role: str, text: str) -> None:
        entry = {"t": time.strftime("%m-%d %H:%M"), "role": role, "text": text[:500]}
        self.entries.append(entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def search(self, query: str, k: int = 3) -> str:
        q = _grams(query)
        if not q:
            return ""
        scored = []
        for e in self.entries:
            t = _grams(e["text"])
            if not t:
                continue
            score = len(q & t) / (len(q) ** 0.5 * len(t) ** 0.5)
            if score > 0.08:                     # 低于这个值基本是巧合
                scored.append((score, e))
        scored.sort(key=lambda x: -x[0])
        return "\n".join(f"- [{e['t']}] {e['text']}" for _, e in scored[:k])


# ---------- 2. 摘要：让模型压缩自己 ----------
def _transcript(msgs: list[dict]) -> str:
    lines = []
    for m in msgs:
        if m.get("tool_calls"):
            calls = ", ".join(f"{c['function']['name']}({c['function']['arguments']})"
                              for c in m["tool_calls"])
            lines.append(f"助手调用工具：{calls}")
            if m.get("content"):
                lines.append(m["content"])
        elif m["role"] == "tool":
            lines.append(f"工具结果：{m['content'][:200]}")
        else:
            lines.append(f"{m['role']}：{m['content'][:400]}")
    return "\n".join(lines)


PROMPT = ("把对话压缩成要点，保留用户的目标、确认过的事实、做出的决定和没做完的事，"
          "去掉寒暄和重复。不超过 200 字，直接给结果。")


def summarize(old: list[dict], prev: str) -> str:
    out = call_llm([
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"已有摘要：{prev or '（无）'}\n\n新增对话：\n{_transcript(old)}"},
    ], [])
    return (out.get("content") or prev).strip()


# ---------- 3. 工作记忆：原文窗口 + 摘要 ----------
@dataclass
class Chat:
    system: str
    summary: str = ""
    messages: list[dict] = field(default_factory=list)

    def add(self, msg: dict) -> None:
        self.messages.append(msg)

    def size(self) -> int:
        return len(self.summary) + sum(len(json.dumps(m, ensure_ascii=False)) for m in self.messages)

    def context(self, recalled: str = "") -> list[dict]:
        system = self.system
        if recalled:
            system += f"\n\n可能相关的历史记录（来自之前的会话，未必准确）：\n{recalled}"
        if self.summary:
            system += f"\n\n更早对话的摘要：\n{self.summary}"
        return [{"role": "system", "content": system}] + self.messages

    def compact(self) -> None:
        while self.size() > BUDGET and len(self.messages) > KEEP_RECENT:
            cut = len(self.messages) - KEEP_RECENT
            # 不能把 tool 消息和它对应的 assistant 切开，否则下一轮直接 400
            while cut > 0 and self.messages[cut]["role"] == "tool":
                cut -= 1
            if cut <= 0:
                break
            old, self.messages = self.messages[:cut], self.messages[cut:]
            self.summary = summarize(old, self.summary)
            print(f"[memory] 压缩 {len(old)} 条旧消息，摘要 {len(self.summary)} 字，"
                  f"剩余 {len(self.messages)} 条原文")

        # 剩下的原文已经不能删了（再删就碰到最近的任务），退一步压缩过长的工具输出
        if self.size() > BUDGET:
            for m in self.messages:
                if m["role"] == "tool" and len(m["content"]) > 600:
                    m["content"] = m["content"][:600] + "…（已截断）"
                    print("[memory] 截断了一条过长的工具输出")


# ---------- 4. 一轮问答 ----------
def run_turn(chat: Chat, store: Store, question: str) -> str:
    recalled = store.search(question)
    if recalled:
        print(f"[memory] 召回 {recalled.count(chr(10)) + 1} 条历史")
    store.add("user", question)
    chat.add({"role": "user", "content": question})

    for step in range(1, MAX_STEPS + 1):
        msg = call_llm(chat.context(recalled), SCHEMA)
        calls = msg.get("tool_calls") or []
        if msg.get("content"):
            print(f"  {msg['content'].strip()}")
        chat.add(msg)

        if not calls:
            final = (msg.get("content") or "").strip()
            store.add("assistant", final[:400])
            chat.compact()
            return final

        for c in calls:
            fn = c["function"]
            print(f"  [tool] {fn['name']}({fn['arguments']})")
            obs = call_tool(fn["name"], fn["arguments"], approve=lambda n, a: True)
            chat.add({"role": "tool", "tool_call_id": c["id"], "content": obs})
            print(f"  [observation] {obs[:120]}")

    chat.compact()
    return "达到最大步数仍未得出结论。"


if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        sys.exit("请先设置环境变量 DEEPSEEK_API_KEY")

    store = Store(MEM_FILE)
    chat = Chat(SYSTEM)

    if sys.argv[1:]:
        print(run_turn(chat, store, " ".join(sys.argv[1:])))
    else:
        print("多轮对话，直接回车退出。")
        while True:
            try:
                q = input("\n你> ").strip()
            except EOFError:
                break
            if not q:
                break
            print("答>", run_turn(chat, store, q))
