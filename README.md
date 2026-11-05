# 从零搭一个 Agent

这个仓库是一条学习路径：一次只讲一个点，每个点配一份能直接跑的 Python 代码。代码刻意保持最小，但每一步都能运行，能改，也能改坏。

## 从哪看起

- [AGENT.md](AGENT.md)：需求、写作约定、章节路线图和进度。每章开写前会先读它。
- `chapters/`：讲义，按主题分章。
- `code/`：与章节一一对应的可运行脚本。

## 章节

| 章 | 主题 | 讲义 | 代码 |
| --- | --- | --- | --- |
| 01 | 最小单 Agent：ReAct 循环 | [chapters/01-react-single-agent.md](chapters/01-react-single-agent.md) | `code/ch01_react.py` |
| 02 | 工具体系：注册表、参数 schema、写操作确认 | [chapters/02-tool-system.md](chapters/02-tool-system.md) | `code/ch02_tools.py` |
| 03 | 原生 function calling；shell 工具与 code agent；MCP | [chapters/03-function-calling.md](chapters/03-function-calling.md) | `code/ch03_function_calling.py`、`code/ch03_code_agent.py`、`code/ch03_mcp.py` |

## 跑起来

```bash
export DEEPSEEK_API_KEY=sk-xxx        # PowerShell: $env:DEEPSEEK_API_KEY="sk-xxx"
python code/ch01_react.py "帮我算一下 (128*7 - 56) / 8"
```

默认接 DeepSeek，换供应商改 `OPENAI_BASE_URL` 和 `MODEL` 两个环境变量即可，第一章第 5 节有一张对照表。

## 环境

Python 3.10 以上。前两章只用标准库，第三章的 MCP 部分也不需要额外依赖——server 和 client 都是手写的，能看清每一条 JSON-RPC 消息。
