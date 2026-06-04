import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

from llm_client import HelloAgentsLLM


class HelloAgentsLLMTest(unittest.TestCase):
    def test_think_can_print_reasoning_and_return_final_content(self):
        """开启 show_reasoning 时，应打印 reasoning 字段，但只返回最终 content。"""
        fake_stream = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content=None, reasoning="先分析问题。"))
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="最终回答。", reasoning=None))
                ]
            ),
        ]

        with patch.dict(
            os.environ,
            {
                "LLM_MODEL_ID": "qwen3:14b",
                "LLM_API_KEY": "ollama",
                "LLM_BASE_URL": "http://127.0.0.1:11435/v1",
            },
            clear=False,
        ):
            with patch("llm_client.OpenAI") as openai_cls:
                openai_instance = MagicMock()
                openai_instance.chat.completions.create.return_value = fake_stream
                openai_cls.return_value = openai_instance

                llm = HelloAgentsLLM()
                output = io.StringIO()
                with redirect_stdout(output):
                    result = llm.think(
                        [{"role": "user", "content": "测试思考和回答"}],
                        show_reasoning=True,
                    )

        self.assertEqual(result, "最终回答。")
        self.assertIn("先分析问题。", output.getvalue())
        self.assertIn("最终回答。", output.getvalue())


if __name__ == "__main__":
    unittest.main()
