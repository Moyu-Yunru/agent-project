"""
Ollama API 调用示例
兼容 OpenAI 格式，用 Python 轻松调用本地大模型
"""

from openai import OpenAI
from openai import OpenAIError

# 通过 SSH 隧道连接到远端 Ollama 服务；本机 11435 已转发到服务器 Ollama 端口。
OLLAMA_BASE_URL = "http://127.0.0.1:11435/v1"

# 使用服务器上已经存在的模型，避免请求不存在的模型导致调用失败。
MODEL_NAME = "qwen3:14b"
TEST_PROMPT = "/no_think\n请只回复四个字：连接成功"
REASONING_OFF = {
    # Ollama 的 OpenAI 兼容接口支持 reasoning_effort/reasoning.effort；
    # 这里显式关闭 thinking，避免 qwen3 只返回 reasoning 而 content 为空。
    "reasoning_effort": "none",
    "reasoning": {"effort": "none"},
}

# 连接到 Ollama 的 OpenAI 兼容接口
client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"  # 本地服务不需要真实 API Key，随便填
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
