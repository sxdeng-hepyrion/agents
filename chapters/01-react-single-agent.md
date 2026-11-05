# 第 1 章　最小单 Agent：ReAct 循环

这一章结束时你会有一个不到两百行的 agent，能真的调用工具回答问题，并且你能指着代码说清楚它每一步在干什么。为了做到这一点，本章只保留三样东西：模型、工具、循环。记忆、规划、多个 agent 互相协作这些都留到后面，因为在一个还看不懂的东西上继续加零件，只会让它更看不懂。

---

## 1. 从一个很窄的函数说起

大模型本身是一个很窄的函数。你给它一段文字，它接下去写一段文字。它读不到你的硬盘，不会真的做除法，也不会记得上一次对话，除非你把上一次的内容重新发给它。所有关于 agent 的讨论，都是从承认这个局限开始的。

要让这个只会说话的东西替你做事，需要补上两样东西。一样是工具，也就是一些真实存在的函数——算数、读文件、发请求——模型可以点名调用它们。另一样是循环，因为模型调用完工具之后，结果必须让它看见，它才能决定下一步；而让它看见的办法只有一个，把结果拼回上下文，再请求一次。

于是整个系统的形状就固定下来了：上下文送进去，模型的输出要么是一个最终答案，要么是一次工具调用。如果是调用，就执行它，把结果写回上下文，再来一轮。

```
用户问题
   │
   ▼
┌──────────────────────────────────┐
│  上下文 messages                 │
│  system + 历史 + 最近一次观察     │
└──────────────┬───────────────────┘
               ▼
          让模型生成"一步"
               │
       ┌───────┴────────┐
   Final Answer      Action + 参数
       │                 │
       ▼                 ▼
     返回给用户      执行真实函数拿到结果
                         │
                         ▼
                  作为 Observation
                  追加回上下文 ─────────┐
                         ▲              │
                         └──────────────┘
```

这三样东西分别对应代码里的 `call_llm`、`TOOLS`，以及 `run` 里面那个循环。后面几章无论加什么，都是在这三样东西上改，不会把它们换掉。

## 2. 为什么要把"想"和"做"拆开

上面这种写法叫 ReAct，来自 2022 年的一篇论文，名字是 Reasoning 和 Acting 的合成。它的出发点是一个很实际的观察：让模型一次性给出复杂问题的答案，出错率高，而且错了你不知道错在哪一步。

拿 `(128*7-56)/8` 举例。如果要求模型直接回答，它需要在内部把三步算术都算完，任何一步滑一下，最后的数字就是错的，而错误被埋在整段回答里，无从检查。如果允许它先调用计算器算 `128*7`，把 896 拿回来，再调用一次算 `(896-56)/8`，那么每一步的中间结果都来自 Python 的真实执行，模型只负责判断下一步该算什么。错误的传播被限制在一步之内，每一步还留下了文字和调用记录。

真正的事实来自 Python 的执行结果，而不是模型的记忆，这是 agent 和纯聊天之间最重要的差别，也是后面所有设计想要保住的东西。

顺便说一句，现在主流模型都支持原生 function calling，那套机制本质上仍然是这个循环，只是把"我要调用计算器，参数是 128*7"从自然语言换成了结构化的 JSON。先用手写文本协议跑通一遍，再去看 JSON schema，才会明白它究竟在解决什么问题，这也是第 3 章的内容。

## 3. 约定一种输出格式

要让循环转起来，模型每一步得按我们认得出来的样子输出。我们只要求四个标记：

```
Thought: 你的思考
Action: 工具名
Action Input: 工具参数

Final Answer: 最终答案
```

`Thought` 是写给人的，程序不解析它，但它有用，因为你能看到模型的判断过程。`Action` 和 `Action Input` 是给程序看的，前者是工具名，后者是参数，约定写在一行里。当模型觉得可以回答用户了，就输 `Final Answer`，循环结束。

这里用纯文本而不是 JSON，是因为纯文本任何模型都能照做，出错的时候也一眼看得出来——模型多写一个词、少一个冒号，你立刻能定位。代价是解析很脆弱，模型稍微不守规矩，正则就抓不到，这一点第 3 章会用结构化输出和原生 function calling 解决。眼下先接受这个脆弱，因为它让整个机制暴露在你眼前。

## 4. 代码

整个 agent 在一个文件里：`code/ch01_react.py`。下面按文件里的顺序读。

### 4.1 模型接入

agent 和模型之间没有 SDK 魔法，就是一次 HTTP POST。`call_llm` 把 `messages` 序列化成 JSON 发到 `/chat/completions`，再取回 `choices[0].message.content`。

```python
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")

# 本地调试可以临时把 key 写在下面，但别提交到 git，也别发到聊天窗口里
API_KEY = os.getenv("DEEPSEEK_API_KEY")

MODEL = os.getenv("MODEL", "deepseek-flash")
MAX_STEPS = 8
WORKDIR = Path.cwd()


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
```

这里刻意只用标准库，没有装 openai 包，为的是让"上下文"这个东西看得见：它就是一个 list，每一轮往后 append 一条。整个 agent 的记忆就是这个 list，没有数据库，也没有藏在某个对象里的隐式状态。

这个 list 的形状也不是随手定的，它是 `/chat/completions` 的协议要求：`messages` 必须是一串消息对象，每条消息两个字段，`role` 和 `content`。跑前面那个默认问题时，第一次请求发出去的 body 是这样的：

```json
{
  "model": "deepseek-flash",
  "temperature": 0,
  "messages": [
    {"role": "system", "content": "你是一个会使用工具的助手……"},
    {"role": "user", "content": "帮我计算 (128*7 - 56) / 8，并说明每一步。"}
  ]
}
```

第二轮会再多两条，模型上一次的回复以 `assistant` 身份追加进去，工具的执行结果以 `user` 身份追加进去。所以一次完整的 ReAct 过程，在请求体里就是一段越来越长的消息列表。

用列表而不是一段拼好的字符串，是因为角色边界本身就是信息。模型需要分清哪句话是用户说的、哪句是自己上一轮说的、哪句来自工具。服务端在把这些消息喂给模型之前，会把每个 `role` 换成对应的特殊标记，拼成一条很长的 prompt。如果你自己把文字拼成整段再送进去，角色信息就降级成了普通文字，模型很容易把自己上一轮写的 `Action: calculator` 当成用户的要求重新执行一遍。

这也解释了 agent 的一个基本事实：它没有持续存在的记忆，每一轮都要把完整历史重发一次。对话越长，这个列表越长，token 越多，钱也越多。第 4 章讲的记忆管理，说到底就是在决定这个列表里该留什么、该丢什么。

至于 `list[dict]` 这个标注，它是省事的写法。`dict` 里面该有哪两个键、`role` 能取哪几个值，从类型上看不出来。写得严谨一点是这样：

```python
from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


def call_llm(messages: list[Message]) -> str:
    ...
```

openai 那个包里的 `ChatCompletionMessageParam` 就是这类定义。这里没有引依赖，所以先用宽松的写法，代价是 `{"role": "assitant", ...}` 这种拼写错误 Python 不会报，要等服务端退回一个 400 才发现。

`temperature` 设成 0，是因为调 agent 的时候需要复现。同样的输入应该得到同样的决策，否则你改了一处代码，分不清效果来自改动还是来自随机性。

默认地址指向 DeepSeek，换供应商只要改环境变量，第 5 节有一张对照。

`API_KEY` 那一行没有给默认值，读不到就直接在 `__main__` 里退出，为的是不拿一个空字符串去请求，换回一个看不出所以然的 401。变量名用 `DEEPSEEK_API_KEY` 而不是通用的 `OPENAI_API_KEY`，是为了和机器上其他工具区分开；换到别家的时候，把这一行的变量名改成对方约定的那个就行。

那段 `try` 不是凑数的。第一次跑这一章的时候，终端上只有一句 `HTTPError: HTTP Error 400: Bad Request`，看不出任何原因，因为 `urlopen` 默认不会把响应体交给你，而 400 的真正原因恰恰写在响应体里。把 `e.read()` 读出来打印之后，屏幕上多了一行 `{"error": {"message": "Model Not Exist"}}`，问题立刻清楚了：模型名写错了。

这也是调试 agent 时值得养成的一个习惯，让错误信息完整地露出来，而不是让它变成一句没有信息量的报错。很多看起来玄学的问题，服务端其实早就把原因写好了，只是被代码吞掉了。所以程序开头还会打印出请求地址、模型名和密钥的最后四位，跑不通的时候先对一遍这三样，比在代码里到处加 print 快得多。

顺着这件事说一句模型名。`MODEL` 的取值必须和供应商的模型清单完全一致，DeepSeek 现在有 `deepseek-flash`、`deepseek-chat` 和 `deepseek-reasoner`，写一个不存在的名字会直接返回 400，而不是自动回落到某个默认模型。

### 4.2 工具

工具在代码里就是普通函数，唯一的额外要求是参数用字符串传进来、返回值也是字符串，因为最后它要以文字形式写回上下文。

```python
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


def calculator(expression: str) -> str:
    """计算数学表达式，如 (12+3)*4"""
    return str(_eval(ast.parse(expression, mode="eval")))
```

`calculator` 里没有用 `eval`。模型输出的字符串属于不可信输入：只要提示词被注入一句 `__import__('os').system(...)`，`eval` 就会真的在你机器上执行命令。这里的做法是把表达式解析成语法树，只放行数字节点和几个算术运算符，遇到别的一律报错。安全上多写的这十几行，是这一章里唯一不能省的部分。

```python
def read_file(path: str) -> str:
    """读取当前工作目录下的文本文件（最多 2000 字）"""
    p = (WORKDIR / path).resolve()
    if WORKDIR not in p.parents and p != WORKDIR:
        return "错误：只能读工作目录内的文件"
    return p.read_text(encoding="utf-8")[:2000]
```

读文件同理。先把路径 `resolve` 成绝对路径，再判断它是否还在工作目录里，这样可以挡掉 `../../Windows/win.ini` 这类越过边界的写法。返回内容截断到 2000 字，因为上下文长度直接换算成费用，一个几兆的日志文件塞进去，这一轮就废了。

两个函数上面那行 `"""..."""` 在这一章里还只是注释。提示词里的工具清单是手工抄进 `SYSTEM` 的，docstring 没有被用起来，这也是为什么改动工具签名时容易漏改说明。第二章会把它变成真的：注册工具时从函数签名和这段文字生成提示词，模型靠它决定该调哪个工具。

最后一行是工具注册表，现在只有两个工具，所以一个字典就够了：

```python
TOOLS = {"calculator": calculator, "read_file": read_file}
```

### 4.3 提示词

`SYSTEM` 这段文字在 agent 里的地位相当于接口文档：上半部分是工具清单，下半部分是输出格式。除了用户的问题和之前的观察，模型每一轮能看到的就只有这些。

```python
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
```

最后那句"不要自己编造 Observation"不是客套。模型很爱顺手写一行 `Observation: 896` 然后接着往下推理，看起来一切正常，其实整个结论建立在自己编的数字上。这句话加上第 4.4 节的解析逻辑，只能压住大部分情况，彻底的办法是把输出变成结构化数据再校验，那是后面的章节。

### 4.4 循环

`run` 是全部逻辑所在，三十行左右。

```python
def run(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
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
```

主流程本身只有三个动作：请求模型，把回复记进历史，然后决定是结束还是调工具。真正花心思的地方是几处保险。

循环有上限 `MAX_STEPS`，因为模型会陷进重复调用同一个工具的死循环，一直转下去。没有这个上限，兜底的就不是代码，而是你的账单。

解析失败不抛异常，而是把一句提醒写回上下文。模型偶尔会不守格式，直接来一句"答案是 42"。这时候让程序崩掉，等于把一个本来能恢复的情况变成了故障；提醒它按格式重写，多数情况下下一轮就对了。

工具执行出错同样不中断，而是把错误信息当成一次观察交给模型。`10/0` 会抛 `ZeroDivisionError`，模型看到"除数不能为零"，通常会换一个式子重算。把异常变成信息，是这个循环能自我修正的前提。

还有一处细节是观察的角色。规范写法应该用 `role=tool`，但不少本地模型和兼容接口对这个角色支持不全，所以这里统一用 `user` 加前缀，先保证能跑，第 3 章再换回规范写法。

## 5. 跑起来

```bash
export DEEPSEEK_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.deepseek.com/v1
export MODEL=deepseek-flash
python code/ch01_react.py "帮我算一下 (128*7 - 56) / 8"
```

Windows 的 PowerShell 里换一种写法：

```powershell
$env:DEEPSEEK_API_KEY="sk-xxx"
$env:OPENAI_BASE_URL="https://api.deepseek.com/v1"
$env:MODEL="deepseek-flash"
python code\ch01_react.py "帮我算一下 (128*7 - 56) / 8"
```

不传参数的时候跑默认问题。需要读文件的问题可以直接问，比如 `python code/ch01_react.py "读一下 notes.txt，把里面的算式算出来"`，前提是那个文件在工作目录里。

换供应商只需要动环境变量：

| 供应商 | `OPENAI_BASE_URL` | `MODEL` |
| --- | --- | --- |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-flash`、`deepseek-chat`、`deepseek-reasoner` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| 本地 Ollama | `http://localhost:11434/v1` | `qwen2.5:7b`，密钥随便填 |

换供应商时除了地址和模型名，还要把读密钥的变量名对上，代码里读的是 `DEEPSEEK_API_KEY`，用 OpenAI 的时候要么导出这个名字，要么把那行改成 `OPENAI_API_KEY`。

跑不通的时候，先看程序开头打印的那三行，再对一下 HTTP 状态码。401 说明密钥无效或者没生效，404 多半是地址写错了（少了或多了一个 `/v1`），400 表示这次请求本身不被接受，模型名不存在、参数不被支持都属于这一类，具体原因就在服务端返回的那段 JSON 里。碰到 400 不要猜，先把返回内容读一遍。

如果不想每次开终端都设环境变量，也可以把 key 填成 `API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-xxx")`，本地调试这样最省事，但那个文件就别再提交到 git 或者发给别人了。

## 6. 一次运行的样子

拿默认问题跑一遍，终端里是这样：

```
URL:   https://api.deepseek.com/v1/chat/completions
MODEL: deepseek-flash
KEY:   ...x9z2（35 个字符）
问题:  帮我计算 (128*7 - 56) / 8，并说明每一步。

--- step 1 ---
Thought: 我需要先计算括号内的乘法，再减去 56，最后除以 8。用计算器确保结果准确。
Action: calculator
Action Input: (128*7 - 56) / 8
[tool] calculator('(128*7 - 56) / 8')
[observation] 105.0

--- step 2 ---
Thought: 计算器已给出结果 105.0，我可以据此逐步说明计算过程。
Final Answer: 计算过程如下：

1. 先算括号内的乘法：128 × 7 = 896
2. 再算括号内的减法：896 − 56 = 840
3. 最后除以 8：840 ÷ 8 = 105

所以 (128 × 7 − 56) ÷ 8 = 105。
```

这次模型把整个表达式一次交给计算器，第二步只是在解释，没有再调用工具。它也可能走另一条路：先算 `128*7` 拿回 896，再算 `(896-56)/8`。两种都算正常的 ReAct，这个循环约束的是"决定、执行、观察"的顺序，并没有规定一步必须做多少。

不过这段输出里有个地方值得停下来看。真正经过计算器的只有 `105.0` 这一个数，前面那两个中间结果 896 和 840 是模型自己在文字里写出来的，如果它算错了，这段解释就是错的，而循环不会发现。更稳妥的用法是让它每一步都调用工具、每个中间结果都来自 Python，代价是多几轮调用。ReAct 能不能靠得住，取决于模型愿不愿意把计算交给工具，而这件事没法只靠提示词保证，所以后面的章节会从结构化输出和结果校验上补。

`[observation] 105.0` 这一类行还有一个用处，它让你看见模型到底调用了什么、拿到了什么。没有这行日志，你只能看到一段像模像样的推理，分不清哪些数字是算出来的，哪些是编出来的。

如果模型第一轮没按格式写，你会看到循环把它拉回来：

```
--- step 1 ---
我觉得答案是 42。

--- step 2 ---
Thought: 我应该用计算器验证一下。
Action: calculator
Action Input: 6*7
[observation] 42
```

第一轮那段回复进了上下文，后面跟着一句"格式不对：必须包含 Action/Action Input，或者 Final Answer。"，模型第二轮就改了写法。这种靠反馈纠偏的做法，本质上是让模型自己修自己的输出，代价是多花一次调用。

## 7. 现在这个版本做不到什么

它只有一个固定不变的提示词，聊上十几轮之后，历史会把上下文窗口塞满，然后开始报错或者遗忘。它没有规划能力，遇到"先查资料再写报告"这种任务，往往走一步看一步，走到一半偏题。它也没法判断自己是不是真的完成了任务，模型说"好了"，程序就信了。工具多了以后，`if name in TOOLS` 加提示词里的一行说明也会不够用，因为模型不知道参数该是什么类型，也没有地方声明哪些操作需要人确认。

这些都是有意留下的口子，每一处对应后面的一章：第 2 章把工具系统做扎实，第 3 章换成结构化调用，第 4 章处理记忆，第 5 章讲规划，第 6 章讲反思，第 7 章让多个 agent 协作，第 8、9 章补上工程和评测。你现在手里的是一个能看懂、能改、也能改坏的东西，这比一个跑得起来但说不清原理的框架有用得多。

## 8. 动手改一改

建议至少做前两个，改动都很小，但能让你对"加一个工具要动几处"形成肌肉记忆。

1. 加一个 `now()` 工具，返回当前时间，然后问它"现在几点了"。需要改的地方只有三处：写函数、加进 `TOOLS`、在 `SYSTEM` 里加一行说明。
2. 把 `MAX_STEPS` 改成 2，问一个需要三步才能算完的问题，看它怎么在步数用尽时收场。
3. 把 `SYSTEM` 里的 `Action:` 故意改成 `Act:`，观察纠正逻辑能不能把模型拉回正轨。
4. 试试让它读 `../../Windows/win.ini`，确认路径检查生效；再让它算 `__import__('os')`，确认 `ast` 白名单拦得住。
5. 想一想：如果模型在 `Action Input` 里写了一段多行的 JSON，`arg.group(1).splitlines()[0]` 会截成什么？这就是为什么后面的章节要把参数变成结构化数据。

## 9. 下一章

下一章把工具系统做扎实，要回答的问题是：工具多起来之后，怎么让模型准确知道每个工具收什么参数、返回什么；工具描述怎么写才不会被用错；删文件、发邮件这类有副作用的操作，应该在哪里设一道人工确认；以及能不能在一次模型回复里同时调用多个工具。
