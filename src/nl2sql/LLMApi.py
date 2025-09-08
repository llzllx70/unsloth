
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
            # model='qwen-max',
            model='qwen3-235b-a22b',
            messages=messages,
            temperature=0.25,
            top_p=0.2,
            extra_body={"enable_thinking": False},
        )

        return response.choices[0].message.content

    def reasoning(self, messages):

        response = self.client.chat.completions.create(
            model='qwen3-235b-a22b',
            messages=messages,
            temperature=0.7,
            top_p=0.6,
            extra_body={"enable_thinking": True},
            # stream=True,
            # stream_options={
            #     "include_usage": True
            # }, 
        )

        is_answering = False  # 是否进入回复阶段

        reasoning_content, answer_content = '', ''

        for chunk in response:

            if not chunk.choices:
                print("\n" + "=" * 20 + "Token 消耗" + "=" * 20 + "\n")
                print(chunk.usage)
                continue

            delta = chunk.choices[0].delta

            # 只收集思考内容
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                if not is_answering:
                    print(delta.reasoning_content, end="", flush=True)
                reasoning_content += delta.reasoning_content

            # 收到content，开始进行回复
            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                print(delta.content, end="", flush=True)
                answer_content += delta.content

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

        【要求如下】 
        1. 回答包括推理过程和最终结论，格式为：<REASONING>推理过程</REASONING>\n<SOLUTION>最终结论</SOLUTION>，要保证REASONING和SOLUTION标签成对出现
        2. 推理过程要列出所有和问题相关的招生信息，如专业，分数，最低位次号等，对符合条件的所有专业都要输出，不要遗漏
        3. 所有推理和结论都要可信，对于无法回复的问题，返回完整格式: <REASONING>信息不全，无法给出建议</REASONING>\n<SOLUTION>请补充信息再提问</SOLUTION>

        【用户问题回答示例: "浙江530分可以选择什么专业？"】
        <REASONING>根据历年信息，浙江2023年最低分小于530分的专业有：专业1(最低分xxx分)，专业2（最低分xxx分）...; 浙江2024年最低分小于530分的专业有：专业1(最低分xxx分)...; </REASONING>
        <SOLUTION>可以选择的专业有：专业A(最低分525分)，专业B(最低分510分)</SOLUTION>

        【用户问题示例回答: "啥专业好"】
        <REASONING>信息不全，无法给出建议</REASONING>
        <SOLUTION>请补充信息再提问</SOLUTION>

        【用户问题】
        {query}
        """

        # 3. 最终结论要简洁明了，尽可能给出结论性的建议, 不做过多解释
        # 4. 如果最低分和最低位次号有冲突，以最低分为依据
        # 5. 如果用户咨询的是2025年的信息，则只能依据2024年及以前的历年信息进行推理

        messages = [
            {"role": "system", "content": "你是浙江树人学院高考咨询助手。"},
            {"role": "user", "content": prompt_}
        ]

        # return self.reasoning(messages)
        return self.chat(messages)
