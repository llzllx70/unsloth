
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
        DataFrame 列: [年份, 省份, 层次, 类别, 专业, 最低分]
        请注意不同省份在层次和类别上不同的划分：
        1. 浙江分本科、专科、专升本、中升本 共4个层次，本科和专科都为普通类，专升本分文史类、理工类、经管类，中升本分外贸类、建筑类
        2. 江苏、福建、湖南：类别分物理类、历史类两类
        3. 河南、广西、贵州、四川、湖北、山西、云南、甘薯、安徽、陕西: 类别分理工、文史两类

        请注意用户的查询意图：
        1. 如果用户咨询的是报考建议，应该检索出历年的信息
        2. 如果用户进行信息查询，可给出具体的检索条件
        3. 如何输入没有指明省份，请不要随意设置

        用户查询: {query}

        请输出合法的 DataFrame.query 语句（只输出语句，不要多余解释, 不要其他格式）。

        正确示例：(省份 == '江苏') & (类别 == '物理类') & (最低分 <= 483)
        错误示例: `df.query("最低分 != ''")`
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
