
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

    def other_tasks(self, df):

        samples = []

        for _ in range(10000):

            row = df.sample(1).iloc[0]

            # 随机选择任务类型
            task_type = random.choice(["字段提取", "条件筛选", "行描述", "比较"])
            
            if task_type == "字段提取":
                # 最低分或最高分提取
                field = random.choice(["最低分", "最高分", "平均分", "最低位次号", "录取数", "省控线"])
                input_text = f"{row['年份']}年{row['省份']}{row['层次']}{row['类别']}{row['专业']}的{field}是多少？答案："

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
                score = random.randint(row["最低分"] - 5, row["平均分"] + 5)
                input_text = f"请列出{row['年份']}年{row['省份']}{row['层次']}{row['类别']}最低分不超过{score}分的专业："
                filtered = df[
                    (df["省份"] == row["省份"]) &
                    (df["层次"] == row["层次"]) &
                    (df["类别"] == row["类别"]) &
                    (df["最低分"] <= score)
                ]
                # 拼接"专业(最低分)"的格式
                target_text = ", ".join(
                    [f"{major}({score_}分)" for major, score_ in zip(filtered["专业"], filtered["最低分"])]
                )

            elif task_type == "行描述":
                input_text = f"请描述{row['年份']}年{row['省份']}{row['层次']}{row['类别']}{row['专业']}的录取情况："
                target_text = (f"{row['专业']}专业{row['年份']}年{row['层次']}{row['类别']}最低分{row['最低分']}，"
                            f"平均分{row['平均分']}，最高分{row['最高分']}，录取数{row['录取数']}，省控线{row['省控线']}")

            elif task_type == "比较":
                # 过滤同一年份、同一省份、且不同专业
                same_group = df[
                    (df["年份"] == row["年份"]) &
                    (df["省份"] == row["省份"]) &
                    (df["专业"] != row["专业"])
                ]
                
                # 如果没有其他专业可以选，就跳过
                if same_group.empty:
                    continue
                
                other_row = same_group.sample(1).iloc[0]

                input_text = f"比较{row['年份']}年{row['省份']}{row['专业']}和{other_row['专业']}的最低分："
                target_text = (
                    f"{row['专业']}最低分{row['最低分']}分，"
                    f"{other_row['专业']}最低分{other_row['最低分']}分。"
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
