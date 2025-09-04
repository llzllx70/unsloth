

import itertools

from src.dataset.BaseDataset import *

class PretrainDataset(BaseDataset):
    
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer, 
            flag="pretrain",
            origin_dataset_file="data/2022_23_24年浙江树人学院各省份录取情况.xlsx"
        )

    def add_whole_row_dataset(self, dataset_):

        def f(e):
            prefix = f'浙江树人学院{e["年份"]}年{e["省份"]}{e["层次"]}{e["类别"]}{e["专业"]}录取情况：'
            return {
                "text": self.row_info(prefix, e),
                "prefix": prefix
            }

        return dataset_.map(f, remove_columns=dataset_.column_names)

    def shuffle_fields(self, row):

        """生成随机排列的 year/province/level/category 拼接字符串"""

        fields = [
            ("year", row["年份"], "{year}年"),
            ("province", row["省份"], "{province}"),
            ("level", row["层次"], "{level}"),
            ("category", row["类别"], "{category}")
        ]
        # 随机选择一种排列
        chosen = random.choice(list(itertools.permutations(fields)))
        # 按顺序拼接
        return "".join([
            fmt.format(year=val, province=val, level=val, category=val)
            for _, val, fmt in chosen
        ])

    def other_tasks(self, df):

        samples = []

        for _ in range(10000):

            row = df.sample(1).iloc[0]

            # 随机选择任务类型
            task_type = random.choice(["字段提取", "条件筛选", "行描述", "比较"])
            
            if task_type == "字段提取":

                field = random.choice(["最低分", "最高分", "平均分", "最低位次号", "录取数", "省控线"])
                prefix_text = self.shuffle_fields(row)

                templates = [
                    "{prefix_text}{major}的{field}是多少？答案：",
                    "请问{prefix_text}{major}{field}？答案：",
                    "帮我查一下{prefix_text}{major}{field}是多少？答案：",
                ]

                input_text = random.choice(templates).format(
                    prefix_text=prefix_text, major=row["专业"], field=field
                )

                # 根据字段动态选择单位
                if field in ["最低分", "最高分", "平均分", "省控线"]:
                    unit = "分"
                elif field in ["录取数"]:
                    unit = "人"
                elif field in ["最低位次号"]:
                    unit = "名"
                else:
                    unit = ""
                
                target_text = f"{row[field]}{unit}"

            elif task_type == "条件筛选":

                score = random.randint(row["最低分"] - 10, row["平均分"] + 10)
                prefix_text = self.shuffle_fields(row)

                templates = [
                    "请列出{prefix_text}最低分不超过{score}分的专业：",
                    "{prefix_text}哪些专业最低分在{score}分以内？",
                    "列出{prefix_text}所有录取最低分≤{score}的专业。",
                    "想知道{prefix_text}录取分数线在{score}分以下的专业。",
                ]
                input_text = random.choice(templates).format(prefix_text=prefix_text, score=score)

                filtered = df[
                    (df["年份"] == row["年份"]) &
                    (df["省份"] == row["省份"]) &
                    (df["层次"] == row["层次"]) &
                    (df["类别"] == row["类别"]) &
                    (df["最低分"] <= score)
                ]
                # 拼接"专业(最低分)"的格式
                if filtered.empty:
                    target_text = "无符合条件的专业"
    
                else:
                    target_text = ", ".join(
                        [f"{major}({score_}分)" for major, score_ in zip(filtered["专业"], filtered["最低分"])]
                    )

            elif task_type == "行描述":

                prefix_text = self.shuffle_fields(row)

                templates = [
                    "请描述{prefix_text}{major}的录取情况：",
                    "{prefix_text}{major}的招生录取详情？",
                    "帮我介绍下{prefix_text}{major}的录取情况。",
                ]
                
                input_text = random.choice(templates).format(prefix_text=prefix_text, major=row["专业"])

                target_text = (
                    f"{prefix_text}{major(row)}，{min_score(row)}，"
                    f"{avg_score(row)}，{max_score(row)}，{admission_num(row)}，{control_line(row)}，{min_rank(row)}。"
                )

            elif task_type == "比较":
                # 过滤同一年份、同一省份、类别且不同专业
                same_group = df[
                    (df["年份"] == row["年份"]) &
                    (df["省份"] == row["省份"]) &
                    (df["层次"] == row["层次"]) &
                    (df["类别"] == row["类别"]) &
                    (df["专业"] != row["专业"])
                ]
                
                # 如果没有其他专业可以选，就跳过
                if same_group.empty:
                    continue
                
                prefix_text = self.shuffle_fields(row)

                templates = [
                    "比较{prefix_text}{major1}和{major2}的最低分：",
                    "{prefix_text}{major1} vs {major2}，哪个最低分高？",
                    "请对比一下{prefix_text}{major1}和{major2}的录取最低分。",
                ]

                other_row = same_group.sample(1).iloc[0]
                input_text = random.choice(templates).format(
                    prefix_text=prefix_text, major1=row["专业"], major2=other_row["专业"]
                )

                target_text = (
                    f"{major(row)}{min_score(row)}，"
                    f"{major(other_row)}{min_score(other_row)}。"
                )

            samples.append({
                "text": f'{input_text}{target_text}',
                "prefix": input_text
            })

        return self.split(Dataset.from_list(samples), test_size=0.05)
        
    def build_dataset(self):

        tr = self.add_whole_row_dataset(dataset_=self.origin_dataset)
        tr1, te1 = self.other_tasks(self.origin_df)

        self.save([tr, tr1], self.train_file)
        self.save([te1], self.test_file)
