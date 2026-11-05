"""第 5 章：检索。切分 → BM25 召回 → LLM 重排 → 组装回答。

语料就是本仓库的讲义，所以问"MCP 的握手流程是什么"能真的找到答案。

运行：
    python ch05_retrieval.py "MCP 的握手流程是什么"
    python ch05_retrieval.py --no-rerank "MCP 的握手流程是什么"   # 看重排前后的差别
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

from ch03_function_calling import call_llm

CORPUS = [Path("AGENT.md"), *sorted(Path("chapters").glob("*.md")), Path("README.md")]
# 不索引本章自己：示例问题会出现在本章原文里，那样检索会命中"引用这个问题的段落"，
# 而不是真正回答它的段落。做文档检索时这类自指噪声很常见。
EXCLUDE = {"05-retrieval.md"}
TOP_K = 12          # 召回多少条
KEEP = 4            # 重排后留给回答的条数


# ---------- 1. 切分：按标题切，太长再按段落切 ----------
def _pack(paras: list[str], max_chars: int) -> list[str]:
    out, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) > max_chars:
            out.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        out.append(buf)
    return out


def build_chunks(max_chars: int = 800) -> list[dict]:
    chunks = []
    for path in CORPUS:
        if not path.exists() or path.name in EXCLUDE:
            continue
        text = path.read_text(encoding="utf-8")
        for part in re.split(r"(?m)^## ", text):
            if not part.strip():
                continue
            lines = part.strip().splitlines()
            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            for piece in _pack([p for p in body.split("\n\n") if p.strip()], max_chars):
                chunks.append({
                    "doc": path.name,
                    "title": title,
                    "text": piece,
                })
    return chunks


# ---------- 2. 稀疏检索：BM25 ----------
def tokenize(s: str) -> list[str]:
    """英文按词、中文按二元组。够用，且不需要分词依赖。"""
    toks: list[str] = []
    for m in re.finditer(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", s.lower()):
        w = m.group()
        if w[0].isascii():
            toks.append(w)
        else:
            toks.extend(w[i:i + 2] for i in range(max(0, len(w) - 1))) or toks.append(w)
    return toks


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(c.values()) for c in self.tfs]
        self.avgdl = sum(self.lens) / max(1, len(docs))
        self.df: Counter = Counter()
        for c in self.tfs:
            self.df.update(c.keys())
        self.N = len(docs)

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1)

    def score(self, query: str, i: int) -> float:
        tf, dl = self.tfs[i], self.lens[i] or 1
        s = 0.0
        for t in set(tokenize(query)):
            f = tf.get(t, 0)
            if not f:
                continue
            s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def top(self, query: str, k: int) -> list[tuple[float, int]]:
        scored = [(self.score(query, i), i) for i in range(self.N)]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(reverse=True)
        return scored[:k]


# ---------- 3. 重排：让模型看 query 和候选一起 ----------
RERANK_PROMPT = """你是检索重排器。根据问题，从候选里挑出真正能回答它的条目，按相关度从高到低排序，最多 5 条。

只输出一个 JSON 数组，元素是候选编号，例如 [3, 1, 7]。没有相关条目就输出 []。

问题：{q}

候选：
{cands}"""


def rerank(question: str, cands: list[dict]) -> list[int]:
    listing = "\n".join(f"[{i}] {c['title']}｜{c['text'][:280]}" for i, c in enumerate(cands))
    out = call_llm([
        {"role": "system", "content": "你是一个严谨的检索重排器，只输出 JSON 数组。"},
        {"role": "user", "content": RERANK_PROMPT.format(q=question, cands=listing)},
    ], [])
    raw = out.get("content") or ""
    m = re.search(r"\[[\d,\s]*\]", raw)
    if not m:
        return list(range(len(cands)))          # 解析失败就保持原顺序
    try:
        order = [int(x) for x in json.loads(m.group())]
    except json.JSONDecodeError:
        return list(range(len(cands)))
    return [i for i in order if 0 <= i < len(cands)] or list(range(len(cands)))


# ---------- 4. 组装：只给选中的几条，要求引用编号 ----------
ANSWER_PROMPT = """根据下面的资料回答问题。只使用资料里的内容，不要补充资料外的知识。
如果资料不足以回答，就直接说"资料里没有提到"，并说明缺什么。
引用时在句末标出来源编号，例如 [2]。

问题：{q}

资料：
{ctx}"""


def answer(question: str, picked: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i + 1}] 《{c['doc']}》{c['title']}\n{c['text'][:700]}"
                      for i, c in enumerate(picked))
    out = call_llm([
        {"role": "system", "content": "你是一个只依据给定资料作答的助手。"},
        {"role": "user", "content": ANSWER_PROMPT.format(q=question, ctx=ctx)},
    ], [])
    return (out.get("content") or "").strip()


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    no_rerank = "--no-rerank" in sys.argv
    question = " ".join(args) or "MCP 的握手流程是什么"

    if not os.getenv("DEEPSEEK_API_KEY"):
        sys.exit("请先设置环境变量 DEEPSEEK_API_KEY")

    chunks = build_chunks()
    bm25 = BM25([c["text"] for c in chunks])
    hits = bm25.top(question, TOP_K)
    cands = [chunks[i] for _, i in hits]

    print(f"语料 {len({c['doc'] for c in chunks})} 个文件 → {len(chunks)} 个块")
    print(f"问题：{question}\n")
    print("召回（BM25）：")
    for score, i in hits[:8]:
        print(f"  {score:6.2f}  {chunks[i]['doc']} · {chunks[i]['title'][:40]}")
    if not cands:
        print("没有召回任何内容")
        return

    if no_rerank:
        picked = cands[:KEEP]
    else:
        order = rerank(question, cands)
        picked = [cands[i] for i in order[:KEEP]]
        print("\n重排后：")
        for rank, i in enumerate(order[:8], 1):
            print(f"  {rank}. {cands[i]['doc']} · {cands[i]['title'][:40]}")

    print("\n--- 回答 ---")
    print(answer(question, picked))


if __name__ == "__main__":
    main()
