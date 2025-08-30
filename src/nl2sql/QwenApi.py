
from openai import OpenAI
import dashscope
import random
from src.nl2sql.secret import qwen_key

dashscope.api_key = qwen_key

class QwenApi:

    def __init__(self, model="qwen-max"):
        self.model_name = 'qwen-max'

    def call(self, messages):

        response = dashscope.Generation.call(
            model=self.model_name,
            messages=messages,
            seed=random.randint(1, 10000),
            result_format='message',  # 将返回结果格式设置为 message
        )

        return response.output.choices[0].message.content

    def nl2sql(self, query: str) -> str:

        prompt = f"""
        你是一个浙江树人学院高考查询助手，对用户给出的自然语言，你需要将它转为 pandas DataFrame.query(expr)中正确的expr参数。
        已知DataFrame 列: [年份	省份 层次 类别 专业	计划数 录取数 省控线 最高分	最低分 平均分 最低位次号]
        
        请注意不同省份在层次划分上的不同:
        1. 浙江有本科、专科、专升本、中升本共4个层次
        2. 江西有本科、专科两个层次
        3. 江苏、福建、湖南、河南、广西、贵州、四川、湖北、山西、云南、甘肃、安徽、陕西均只有本科一个层次

        请注意不同省份在类别划分上的不同：
        2. 浙江本科和专科类别都为普通，专升本分文史、理工、经管三类，中升本分外贸、建筑两类
        2. 江苏、福建、湖南分物理、历史两类
        3. 河南、广西、贵州、四川、湖北、山西、云南、甘肃、安徽、陕西分理工、文史两类

        请注意用户的查询意图：
        1. 如果用户咨询的是报考建议，应该检索出历年的信息
        2. 如果用户进行信息查询，需给出具体的检索条件
        3. 如果没有指明省份，默认为浙江
        4. 所有的咨询问题都针对的是浙江树人学院

        请输出可作为DataFrame.query(expr)参数expr的查询语句，不要多余解释, 不要其他格式:

        【示例1: 江苏，物生地，483分，可以选择什么专业】
        正确：(省份 == '江苏') & (类别 == '物理') & (最低分 <= 483)
        错误: `df.query("最低分 != ''")`

        【示例2: 最低录取分数线或者位次是多少】
        正确: (省份 == '浙江') & ((最低分 != '') | (最低位次号 != ''))
        错误: ((省份 == '浙江') & ((最低分 != '')) | (最低位次号 != '')

        【示例3: 530分文科有机会吗】
        正确: (省份 == '浙江') & (类别 == '普通') & (最低分 <= 530)
        错误: (省份 == '浙江') & (类别 == '文史') & (最低分 < 530)

        用户查询: {query}
        """

        messages = [
            {"role": "user", "content": prompt}
        ]

        return self.call(messages)

    def answer(self, query, md):

        prompt_ = f"""
        已知信息:
        1. 今年为2025年
        2. 历年报考信息:【{md}】

        依据上面的信息回答用户报考问题：【{query}】 
        要求如下： 
        1. 将推理过程置入<REASONING>和</REASONING>内 
        2. 再将答案置入 <SOLUTION>和</SOLUTION>内
        """

        messages = [
            {"role": "system", "content": "你是浙江树人学院高考查询助手。"},
            {"role": "user", "content": prompt_}
        ]

        return self.call(messages)
