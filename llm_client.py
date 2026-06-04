"""
运行前请先保持 SSH 隧道打开：
ssh -N -L 11435:127.0.0.1:11434 server1
"""
import os
from typing import Dict
from typing import List
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


# 加载项目根目录 .env 中的模型服务配置。
load_dotenv()


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


class HelloAgentsLLM:
    """
    为 "Hello Agents" 定制的 LLM 客户端。

    它封装了 OpenAI 兼容接口的初始化、配置读取和流式响应处理，
    让智能体主逻辑只需要关心 messages 输入和模型返回文本。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        apiKey: Optional[str] = None,
        baseUrl: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """
        初始化客户端。

        参数优先级：
        1. 调用方显式传入的参数；
        2. 推荐的新环境变量：LLM_MODEL_ID、LLM_API_KEY、LLM_BASE_URL、LLM_TIMEOUT；
        3. 兼容当前项目已有环境变量：MODEL_NAME、OPENAI_API_KEY、OPENAI_BASE_URL。
        """
        self.model = model or os.getenv("LLM_MODEL_ID") or os.getenv("MODEL_NAME")
        api_key = apiKey or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = baseUrl or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        timeout_value = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, api_key, base_url]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在 .env 文件中定义。")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_value)

    def think(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        show_reasoning: bool = False,
        reasoning_effort: str = "medium",
    ) -> Optional[str]:
        """
        调用大语言模型进行思考，并返回完整响应文本。

        messages 使用 OpenAI Chat Completions 格式，例如：
        [
            {"role": "system", "content": "你是一个 Python 老师。"},
            {"role": "user", "content": "写一个快速排序算法"},
        ]

        show_reasoning=True 时会额外打印 qwen3/Ollama 返回的 reasoning/thinking 字段；
        返回值仍然只包含最终回答 content，方便智能体主逻辑继续处理。
        """
        print(f"正在调用 {self.model} 模型...")
        try:
            extra_body = None
            if show_reasoning:
                # Ollama 的 OpenAI 兼容接口用该参数控制 thinking 模型的思考强度。
                extra_body = {
                    "reasoning_effort": reasoning_effort,
                    "reasoning": {"effort": reasoning_effort},
                }

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
                extra_body=extra_body,
            )

            # 处理流式响应：每个 chunk 只包含一小段增量文本，需要边打印边收集。
            print("大语言模型响应成功:")
            if show_reasoning:
                print("--- 模型思考 ---")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = read_extra_field(delta, "reasoning", "reasoning_content", "thinking")
                content = delta.content or ""

                if show_reasoning and reasoning:
                    print(reasoning, end="", flush=True)

                if content:
                    collected_content.append(content)

            if show_reasoning:
                print("\n--- 最终回答 ---")
            print("".join(collected_content), end="", flush=True)
            print()
            return "".join(collected_content)

        except Exception as error:
            print(f"调用 LLM API 时发生错误: {error}")
            return None


if __name__ == "__main__":
    try:
        llm_client = HelloAgentsLLM()
        example_messages = [
            {"role": "system", "content": "你是一个精通心理学、哲学的助手."},
            {"role": "user", "content": "简要概括一下虚无主义"},
        ]

        print("--- 调用 LLM ---")
        response_text = llm_client.think(example_messages, show_reasoning=True)
        if response_text:
            print("\n--- 完整模型响应 ---")
            print(response_text)

    except ValueError as error:
        print(error)
