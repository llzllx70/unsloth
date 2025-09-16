
from openai import OpenAI
import random
from src.nl2sql.secret import qwen_key

class LLMApi:

    def __init__(self):

        self.client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=qwen_key
        )

    def chat(self, messages):

        response = self.client.chat.completions.create(
            model='qwen-max',
            # model='qwen3-235b-a22b',
            messages=messages,
            temperature=0.1,
            top_p=0.2,
            extra_body={"enable_thinking": False},
        )

        return response.choices[0].message.content

    def reasoning(self, messages):

        response = self.client.chat.completions.create(
            model='deepseek-r1',
            messages=messages,
        )

        reasoning_content = response.choices[0].message.reasoning_content
        answer_content = response.choices[0].message.content

        return reasoning_content, answer_content

    def nl2sql(self, query: str) -> str:

        # 已知DataFrame 列: [年份	省份 层次 类别 专业	计划数 录取数 省控线 最高分	最低分 平均分 最低位次号]

        prompt = f"""
        你是一个浙江树人学院高考查询助手，对用户给出的自然语言，你需要将它转为 pandas DataFrame.query(expr)中正确的expr参数。
        已知DataFrame 列: [年份	省份 层次 类别]，请针对这些列进行查询。
        
        请注意不同省份在层次划分上的不同:
        1. 浙江有本科、专科、专升本、中升本共4个层次
        2. 江西有本科、专科两个层次
        3. 江苏、福建、湖南、河南、广西、贵州、四川、湖北、山西、云南、甘肃、安徽、陕西均只有本科一个层次

        请注意不同省份在类别划分上的不同：
        1. 浙江本科和专科类别都为普通，专升本分文史、理工、经管三类，中升本分外贸、建筑两类
        2. 江苏、福建、湖南、江西、河南、广西、贵州、四川、湖北、山西、云南、甘肃、安徽、陕西分理工、文史两类
        3. 没有物理和历史, 正确的为理工和文史

        请注意用户的查询意图：
        1. 如果用户咨询的是报考建议，应该检索出历年的信息
        2. 查询语句中，只需要返回年份, 省份, 层次, 类别的相关查询，尤其要注意不同省份的层次和类别划分
        3. 如果没有指明省份，默认为浙江
        4. 如果用户查询没有指定年份，不要加年份条件
        5. 所有的咨询问题都针对的是浙江树人学院

        【示例1: 江苏，物生地，483分，可以选择什么专业】
        正确：(省份 == '江苏') & (类别 == '理工')
        错误: `df.query("最低分 != ''")`
        错误原因：非查询语句

        【示例2: 530分文科有机会吗】
        正确: (省份 == '浙江') & (类别 == '普通')
        错误: (省份 == '浙江') & (类别 == '文史')
        错误原因：类别不对

        【示例3: 湖北考生历史类444分有希望考进树人学院吗】
        正确: (省份 == '湖北') & (类别 == '文史')
        错误: (省份 == '湖北') & (类别 == '历史')
        错误原因：类别不对，没有历史

        【示例4: 江西今年理科407】
        正确: (省份 == '江西') & (类别 == '理工')
        错误: (省份 == '江西') & (类别 == '本科')
        错误原因: 层次和类别混淆

        【示例5：浙江物化504分能进计算机专业吗】
        正确: (省份 == '浙江') & (类别 == '普通') 
        错误: (省份 == '浙江') & (类别 == '普通') & (专业 == '计算机') & (最低分 <= 504)
        错误原因：包含多余检索项

        请输出可作为DataFrame.query(expr)参数expr的查询语句，不要多余解释, 不要其他格式
        用户查询: {query}
        """

        messages = [
            {"role": "user", "content": prompt}
        ]

        return self.chat(messages)

    def answer(self, query, md):

        prompt_ = f"""你需要依据下面提供的信息和要求回答用户问题 
        【历年报考信息】
        {md}

        【要求】
        1. 你只可依据上述提供信息回答问题，不要依据任何额外信息或编造信息
        2. 一定要保证引用信息的完整性和正确性
        3. 如果最低分和最低位次号都有提供，以最低分为主要依据，因为位次号是估计值，不如最低分准确
        
        【用户问题】
        {query}
        """

        messages = [
            {"role": "system", "content": "你是浙江树人学院高考咨询助手。"},
            {"role": "user", "content": prompt_}
        ]

        return self.reasoning(messages)
        # return self.chat(messages)

    def reward_score(self, reasoning, solution, reasoningi, solutioni):

        prompt_ = f"""你需要对比模型生成文本基于参考文本的语义相似度，并进行评分
        【评分要求】
        1. 参考文本是正确答案，你需要据此对模型生成文本的正确性进行打分，满分10分，最终分数不低于0分
        2. 参考文本和模型生成文本均包含推理过程和答案，分别用reasoning和solution标识
        3. 模型生成文本若存在明确的事实错误, 每处错误扣2分
        4. 模型生成文本若和参考文本若存在语义不一致，每处扣1分
        5. 对于无法判断的内容，不要随意扣分

        【返回格式包括评分依据和最终得分】
        <judge>评分依据xxx</judge>
        <score>最终得分</score

        【示例】
        <judge>评分依据：
        问题1. 在参考文本中提到2024年浙江工商管理最低分为552分，但是在所给文本却是540分，存在事实错误，扣2分 
        问题2. 参考文本中提到户录取可能性不大，而在所给文本中提到很有希望被录取，存在语义不一致，扣1分
        所以最终得分为: 10-2-1=7分
        </judge>
        <score>7</score

        【参考文本】
        <reasoning>{reasoning}</reasoning>
        <solution>{solution}</solution>

        【模型生成文本】
        <reasoning>{reasoningi}</reasoning>
        <solution>{solutioni}</solution>
        """

        messages = [
            {"role": "system", "content": "你是一个评价师 。"},
            {"role": "user", "content": prompt_}
        ]

        return self.chat(messages=messages)

        
        
