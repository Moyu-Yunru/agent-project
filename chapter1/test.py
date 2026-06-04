import os
from dotenv import load_dotenv
load_dotenv()

# 测试天气 API
import requests
response = requests.get("https://wttr.in/Beijing?format=j1")
print("天气API状态:", response.status_code)

# 测试 Tavily API
from tavily import TavilyClient
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
try:
    result = tavily.search("test", search_depth="basic")
    print("Tavily API 连接成功")
except Exception as e:
    print("Tavily API 错误:", e)

# 测试 LLM API - AIHubmix
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
try:
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10,
    )
    print("LLM API 连接成功:", response.choices[0].message.content)
except Exception as e:
    print("LLM API 错误:", e)

# 测试 LLM API - ModelScope（如果您使用的是 ModelScope，请取消注释并替换配置）
# from openai import OpenAI
# client = OpenAI(
#     api_key="your_modelscope_api_key",
#     base_url="https://api-inference.modelscope.cn/v1/"
# )
# try:
#     response = client.chat.completions.create(
#         model="Qwen/Qwen2.5-72B-Instruct",
#         messages=[{"role": "user", "content": "Hello"}],
#         max_tokens=10
#     )
#     print("LLM API 连接成功:", response.choices[0].message.content)
# except Exception as e:
#     print("LLM API 错误:", e)