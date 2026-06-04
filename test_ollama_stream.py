"""
流式测试 qwen3 的“思考过程”和“最终回答”。

运行前请先保持 SSH 隧道打开：
ssh -N -L 11435:127.0.0.1:11434 server1
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError

load_dotenv()


def require_env(name):
    """读取必需环境变量；缺失时尽早给出清晰错误，避免 OpenAI SDK 内部报错不直观。"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"请先在 .env 中配置 {name}")
    return value


# 模型服务配置统一从 .env 读取，切换 OpenAI 官方、本地 Ollama 或第三方兼容服务时只需要改 .env。
OPENAI_BASE_URL = require_env("OPENAI_BASE_URL")
OPENAI_API_KEY = require_env("OPENAI_API_KEY")
MODEL_NAME = require_env("MODEL_NAME")

# 这个输入会被发送到模型。system 用来规定角色，user 用来放你的真实问题。
MESSAGES = [
    {
        "role": "system",
        "content": "你是一个严谨的 Python 老师。先思考，再给出简洁清晰的最终回答。",
    },
    {
        "role": "user",
        "content": "请判断 Python 列表推导式和 for 循环在可读性上的区别，并给一个短例子。",
    },
]

# 让 qwen3 输出可观察的思考字段；如果想关闭思考，可把 effort 改成 "none"。
REASONING_ON = {
    "reasoning_effort": "medium",
    "reasoning": {"effort": "medium"},
}


def read_extra_field(value, *field_names):
    """兼容 OpenAI SDK 对 Ollama 非标准字段的多种承载方式。"""
    for field_name in field_names:
        field_value = getattr(value, field_name, None)
        if field_value:
            return field_value

    if hasattr(value, "model_dump"):
        data = value.model_dump()
        for field_name in field_names:
            field_value = data.get(field_name)
            if field_value:
                return field_value

    model_extra = getattr(value, "model_extra", None) or {}
    for field_name in field_names:
        field_value = model_extra.get(field_name)
        if field_value:
            return field_value

    return None


client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

try:
    print("=== 输入给模型的 messages ===")
    for message in MESSAGES:
        print(f"[{message['role']}] {message['content']}")

    print("\n=== AI 思考过程 reasoning ===")
    printed_reasoning = False

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        temperature=0.2,
        max_tokens=800,
        stream=True,
        extra_body=REASONING_ON,
    )

    answer_chunks = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        reasoning = read_extra_field(delta, "reasoning", "reasoning_content", "thinking")
        content = delta.content

        if reasoning:
            printed_reasoning = True
            print(reasoning, end="", flush=True)

        if content:
            answer_chunks.append(content)

    if not printed_reasoning:
        print("没有收到 reasoning 字段。远端 Ollama 可能把思考隐藏了，或模型模板不返回该字段。")

    print("\n\n=== AI 最终回答 content ===")
    answer = "".join(answer_chunks).strip()
    if answer:
        print(answer)
    else:
        print("没有收到 content 字段。请尝试把 REASONING_ON 的 effort 改成 \"none\"，或更新远端 Ollama/qwen3。")

except OpenAIError as error:
    # 502 表示请求已经到达服务端，但 Ollama 或模型 runner 在处理生成时失败。
    print("\n调用 Ollama 流式接口失败：")
    print(error)
    print("如果错误码是 502，请优先检查远端 Ollama 日志、模型是否成功加载、服务器内存/显存是否不足。")
