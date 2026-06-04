"""
Ollama API 调用示例
兼容 OpenAI 格式，用 Python 轻松调用本地大模型
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
TEST_PROMPT = "/no_think\n请只回复四个字：连接成功"
REASONING_OFF = {
    # Ollama 的 OpenAI 兼容接口支持 reasoning_effort/reasoning.effort；
    # 这里显式关闭 thinking，避免 qwen3 只返回 reasoning 而 content 为空。
    "reasoning_effort": "none",
    "reasoning": {"effort": "none"},
}

# 连接到 Ollama 的 OpenAI 兼容接口
client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

try:
    # 先用很短的请求测试“隧道 -> Ollama -> 模型生成”全链路，避免长回答把问题混到负载/OOM里。
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": TEST_PROMPT}
        ],
        temperature=0,
        max_tokens=64,
        extra_body=REASONING_OFF
    )

    message = response.choices[0].message
    content = message.content
    reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)

    # 打印回复；和流式脚本保持同一行，避免误以为“AI 回复：”后面为空。
    if content:
        print(f"AI 回复：{content}")
    elif reasoning:
        print("AI 回复：")
        print("请求成功，但模型只返回了 reasoning 字段，没有返回正文 content。")
        print("下面打印 reasoning 内容，说明模型确实已经生成了结果：")
        print(reasoning)
    else:
        print("AI 回复：")
        print("请求成功，但模型返回正文为空。")
except OpenAIError as error:
    # 502 表示请求已经到达服务端，但 Ollama 或模型 runner 在处理生成时失败。
    print("调用 Ollama 失败：")
    print(error)
    print("如果错误码是 502，请优先检查远端 Ollama 日志、模型是否成功加载、服务器内存/显存是否不足。")
