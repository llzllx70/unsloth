
from openai import OpenAI
import dashscope
import random
from src.nl2sql.secret import qwen_key

dashscope.api_key = qwen_key

class LLMQueryParser:
    def __init__(self, model="qwen-max"):
        self.model_name = 'qwen-max'

    def parse(self, query: str) -> str:
        prompt = f"""
        你是一个助手，用户给出自然语言查询，你需要将它转为 pandas DataFrame.query 语句。
        DataFrame 列: [年份, 省份, 专业, 类别, 最高分, 最低分, 平均分, 位次号]
        注意点：
        1. 目前类别有物理类和历史类

        用户查询: {query}

        请输出合法的 DataFrame.query 语句（只输出语句，不要多余解释）。
        """

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = dashscope.Generation.call(
            model=self.model_name,
            messages=messages,
            seed=random.randint(1, 10000),
            result_format='message',  # 将返回结果格式设置为 message
        )

        return response.output.choices[0].message.content
