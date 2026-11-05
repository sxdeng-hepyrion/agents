# AGENT.md — 个人 Agent 系统：需求与进度

> 这是本项目的总纲。每次开始新的一节前我会先读它；你提的新要求会追加到最后的「需求变更记录」。

## 1. 目标

从零搭建一个**属于自己的个人 agent 系统**，理解每一个零件，而不是套一个现成框架。

## 2. 学习方式（约定）

- **一次一个点**：你指定一个主题或问题，我只讲这一个点相关的原理 + 最小可运行代码，不铺开讲全貌。
- **不灌输**：不写用不上的抽象层，不提前引入后面的章节才需要的东西。
- **代码**：Python。只保留与当前主题最相关的部分，结构尽量短；但必须能直接跑起来。
- **组织**：一个主题一个章节 `chapters/NN-xxx.md`，章内包含 原理 → 代码 → 怎么跑 → 常见坑 → 练习 → 下一章预告。可运行脚本放 `code/`，与章节一一对应。
- **递进**：后面的章节直接复用前面的代码，只替换需要变的部分。

### 写作风格（重要）

讲义要能顺着读下来，像讲一遍课，而不是把要点敲在墙上。

- 用成段的文字推进，段与段之间有承接关系（因为、所以、不过、于是、换句话说），不要靠一堆短句和清单堆信息。
- 不写"三个细节值得注意""这是调试 agent 的唯一抓手""一句话总结"这类提示语、口号和夸张断言。
- 加粗只留给真正容易看漏的地方，不要每段都加；不用斜体强调。
- 清单只用于天生并列的东西，比如操作步骤、参数表、供应商对照。
- 代码块前后的文字要说明"为什么这么做"和"看到的结果意味着什么"，而不是复述代码。

## 3. 目录结构

```
AGENT.md                  ← 本文件：需求、约定、进度
chapters/                 ← 讲义（md，按章）
  NN-xxx.md
code/                     ← 每章配套的可运行脚本
  ch01_react.py           文本协议的最小 ReAct
  ch02_tools.py           工具注册表与参数校验
  ch03_function_calling.py  原生 function calling 的循环
  ch03_code_agent.py      code agent 的五个工具（view/search/edit/run_command）
  mcp_server.py           MCP server（stdio）
  mcp_client.py           MCP client + 自检
  ch03_mcp.py             用 MCP 工具跑同一套循环
  ch04_memory.py          滑动窗口 + 摘要压缩 + 长期检索
  ch05_retrieval.py       切分 + BM25 + LLM 重排 + 引用作答
```

## 4. 技术选型

| 项 | 选择 | 理由 |
| --- | --- | --- |
| 语言 | Python 3.10+ | — |
| 依赖 | 优先只用标准库 | 少一层遮蔽，看得见真实发生了什么 |
| 模型接入 | OpenAI 兼容的 `/chat/completions` | 换 `base_url` 就能在 DeepSeek / OpenAI / Ollama 之间切换 |
| 框架 | 不用（先手写） | 先理解机制，再看 LangChain 之类才有判断力 |

环境变量约定：

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 密钥（Ollama 可随便填） | 无，必须设置 |
| `OPENAI_BASE_URL` | 接口地址 | `https://api.deepseek.com/v1` |
| `MODEL` | 模型名 | `deepseek-flash` |

实测环境：DeepSeek `deepseek-flash`，Python 3.14（conda 环境 `learning`）。模型名必须和供应商的清单一致，写错会直接 400。

## 5. 章节路线图

顺序会随你的兴趣调整——你说哪个点，就先做哪个。

| 章 | 主题 | 文件 | 状态 |
| --- | --- | --- | --- |
| 01 | 最小单 agent：ReAct 循环（Thought / Action / Observation） | [chapters/01-react-single-agent.md](chapters/01-react-single-agent.md) | ✅ 已完成 |
| 02 | 工具系统：注册表、参数 schema、错误回传、写操作确认 | [chapters/02-tool-system.md](chapters/02-tool-system.md) | ✅ 已完成 |
| 03 | 原生 function calling：tool_calls、tool 角色、并行调用；shell 工具与 code agent 工具集；MCP 接入 | [chapters/03-function-calling.md](chapters/03-function-calling.md) | ✅ 已完成 |
| 04 | 记忆：滑动窗口、摘要压缩、跨会话检索 | [chapters/04-memory.md](chapters/04-memory.md) | ✅ 已完成 |
| 05 | 检索：切分、BM25、embedding、RRF、重排、评估 | [chapters/05-retrieval.md](chapters/05-retrieval.md) | ✅ 已完成 |
| 06 | 规划：Plan-and-Execute / 任务分解 | — | 待开始 |
| 07 | 反思：Self-Refine / 自我检查 | — | 待开始 |
| 08 | 多 agent 协作：Supervisor 与消息传递 | — | 待开始 |
| 09 | 工程化：日志、重试、成本与限流 | — | 待开始 |
| 10 | 评测：怎么知道它变好了 | — | 待开始 |
| 11 | 组装成"个人助理"：入口、持久化、真实工具、安全边界 | — | 待开始 |

## 6. 版本管理

从第 3 章起用 git 管理，仓库在本地，尚未配置远程。

- 每完成一章（或一次实质性修改）提交一次，message 用 `第 N 章：<主题>`；修 bug、改讲义用 `fix:` / `docs:` 前缀。
- `.gitignore` 已经排除了 `__pycache__`、`.tmp/`、`result.txt`、`out.txt` 和 `.env` 一类文件，别把它们提交进去。
- 密钥只从环境变量读，任何情况下都不写进仓库——历史里的密钥很难彻底删掉。
- 换行统一成 LF（`.gitattributes` 里定的），避免以后在不同机器上出现整文件 diff。

以后要接远程仓库，先把空仓库建好，然后：

```bash
git remote add origin git@github.com:<用户名>/<仓库名>.git
git push -u origin main
```

## 7. 需求变更记录

- **2026-11-04** 初始需求：一次讲一个知识点；Python；代码只留最相关内容但要可运行；用 md 分章节组织；从最简单的单 agent ReAct 开始。
- **2026-11-05** 加入第 3 章的 shell 工具、code agent 与 MCP 内容；开始用 git 管理，暂不推远程。
