import json
from datetime import date
from typing import Any, Dict, Optional, Tuple

from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search

# ReAct 提示词：约束模型每轮只返回一个结构化 JSON 对象，方便程序稳定解析。
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。
当前日期：{current_date}

可用工具如下：
{tools}

请严格按照以下 JSON 格式进行回应，并且只输出一个 JSON 对象，不要输出 Markdown、代码块、解释文字或额外字段：

{{
  "thought": "你的思考过程，用于分析问题、拆解任务和规划下一步行动。",
  "action": {{
    "name": "工具名称或 Finish",
    "input": "工具输入，或最终答案"
  }}
}}

字段规则：
- `thought` 必须是字符串，用于简要说明你的分析和下一步计划。
- `action.name` 必须是可用工具名称之一，或 `Finish`。
- `action.input` 必须是字符串；调用工具时填写工具输入，结束任务时填写最终答案。
- 当你收集到足够的信息，能够回答用户的最终问题时，必须使用 `{{"name": "Finish", "input": "最终答案"}}`。
- 如果用户询问“最新”“当前”“现在”等实时问题，必须基于当前日期使用工具检索最新信息。
- 不要在搜索词中凭空加入用户没有提供的过去年份；除非用户明确指定年份。


现在，请开始解决以下问题：
Question: {question}
History: {history}
"""

class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                current_date=date.today().isoformat(),
                tools=tools_desc,
                question=question,
                history=history_str,
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("错误：LLM未能返回有效响应。"); break

            thought, action = self._parse_output(response_text)
            if thought: print(f"🤔 思考: {thought}")
            if not action: print("警告：未能解析出有效的Action，流程终止。"); break

            action_name = action["name"]
            action_input = action["input"]

            if action_name == "Finish":
                # Finish 也是一个结构化 action：input 字段直接承载最终答案。
                final_answer = action_input
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = action_name, action_input
            if not tool_name:
                self.history.append("Observation: 无效的 JSON Action 格式，请检查 action.name。"); continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            tool_function = self.tool_executor.getTool(tool_name)
            observation = tool_function(tool_input) if tool_function else f"错误：未找到名为 '{tool_name}' 的工具。"
            
            print(f"👀 观察: {observation}")
            # 将结构化 action 以 JSON 字符串写入历史，避免下一轮模型看到 Python dict 表示后混淆输出协议。
            self.history.append(f"Action: {json.dumps(action, ensure_ascii=False)}")
            self.history.append(f"Observation: {observation}")

        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
        """
        解析模型返回的结构化 JSON 输出。

        之前的实现依赖 `Thought:`、`Action:` 和方括号等自然语言标记；
        一旦模型多输出解释、复制示例、或工具输入中包含类似标记，正则就容易截错。
        这里改成 JSON 后，字段边界由 JSON 解析器负责处理，字符串里的括号、
        换行和 `Action:` 字样都会被当作普通内容，而不会影响 action 的边界。
        """
        payload = self._load_json_payload(text)
        if payload is None:
            return None, None

        thought = payload.get("thought")
        action = payload.get("action")

        if not isinstance(thought, str):
            thought = None
        else:
            thought = thought.strip()

        if not isinstance(action, dict):
            return thought, None

        action_name = action.get("name")
        action_input = action.get("input", "")
        if not isinstance(action_name, str) or not action_name.strip():
            return thought, None

        # 工具执行器当前接收字符串输入；如果模型偶尔返回数字、列表或对象，
        # 这里统一序列化成 JSON 字符串，既保留原始结构，也避免类型错误中断流程。
        if isinstance(action_input, str):
            normalized_input = action_input.strip()
        else:
            normalized_input = json.dumps(action_input, ensure_ascii=False)

        normalized_name = action_name.strip()
        if normalized_name.lower() == "finish":
            # Finish 是协议里的保留动作，做大小写归一化可以容忍模型偶尔输出 finish/FINISH。
            normalized_name = "Finish"

        normalized_action = {
            "name": normalized_name,
            "input": normalized_input,
        }
        return thought, normalized_action

    def _load_json_payload(self, text: str) -> Optional[Dict[str, Any]]:
        """
        将模型输出转换为 JSON 对象。

        主协议要求模型只输出 JSON；额外兼容 ```json 代码块，是因为很多模型即使被要求
        “只输出 JSON”，仍可能习惯性包一层 Markdown。除此之外保持严格解析，
        让格式错误尽早暴露，避免把自然语言说明误当成可执行指令。
        """
        if not text:
            return None

        json_text = self._strip_json_code_fence(text)
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None
        return payload

    def _strip_json_code_fence(self, text: str) -> str:
        """
        去掉模型常见的 Markdown 代码块外壳。

        这个步骤只做“包装层”清理，不尝试从长篇自然语言中猜测 JSON 起止位置；
        这样可以兼顾容错和安全性：代码块包装能被接受，混杂解释文字则会解析失败。
        """
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)
