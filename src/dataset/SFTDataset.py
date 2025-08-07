from src.dataset.BaseDataset import *

class SFTDataset(BaseDataset):
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, flag="sft")

    def kn_message(self, x):
        """
        知识训练语料
        """
        expected_answer = x["expected_answer"]
        problem = x["problem"]

        return [
            {"role": "system", "content": sft_system_prompt},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": expected_answer},
        ]

    def kn_format_message(self, x):
        """
        知识+格式训练语料
        """
        expected_answer = x["expected_answer"]
        problem = x["problem"]

        # Remove generated <think> and </think>
        thoughts = x["generated_solution"]
        thoughts = thoughts.replace("<think>", "").replace("</think>", "")
        
        # Strip newlines on left and right
        thoughts = thoughts.strip()
        
        # Add our custom formatting
        final_prompt = (
            reasoning_start + thoughts + reasoning_end +
            solution_start + expected_answer + solution_end
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": final_prompt},
        ]

    
    def row_info(self, e):

        return (
            f"浙江省2024年本科{e['专业']}专业的录取计划数为{e['计划数']}人，"
            f"录取数为{e['录取数']}人，省控线为{e['省控线']}分。"
            f"最高分为{e['最高分']}分，最低分为{e['最低分']}分，"
            f"平均分为{e['平均分']}分，最低位次号为{e['最低位次号']}。"
        )

    def add_whole_row_dataset(self, dataset_):

        def f(e):
            problem = f'浙江省2024年本科{e["专业"]}录取情况'

            return {
                "expected_answer": self.row_info(e),
                "problem": problem,
                "generated_solution": f'好的，针对{problem}，我将从{list(dict(e).keys())}这些方面为您提供相关信息。',
            }

        dataset_1 = dataset_.map(f, remove_columns=dataset_.column_names)

        return self.split(dataset_1)

    def add_one_dimension_dataset(self, dataset_):

        def f(e):

            d = random.choice(fields)

            prefix = f'浙江省2024年本科{e["专业"]}专业的{d}'
            problem = f'{prefix}是多少？'
            expected_answer = f'{prefix}是{e[d]}'

            return (
                {
                    "expected_answer": expected_answer,
                    "problem": problem,
                    "generated_solution": f'好的，针对{problem}的问题，可以搜索到如下相关信息{self.row_info(e)}',
                }
            )

        fields = ["计划数", "录取数", "省控线", "最高分", "最低分", "平均分", "最低位次号"]

        dataset_2 = dataset_.map(f, remove_columns=dataset_.column_names)

        return self.split(dataset_2)

    def build_dataset(self):

        tr1, te1 = self.add_whole_row_dataset(dataset_=self.origin_dataset_)
        tr2, te2 = self.add_one_dimension_dataset(dataset_=self.origin_dataset_)

        self.save([tr1, tr2], self.train_file)
        self.save([te1, te2], self.test_file)

    def prepare_dataset(self, dataset_):

        dataset_ = dataset_.to_pandas()[
            ["expected_answer", "problem", "generated_solution"]
        ]

        # pandas to JSON
        dataset_["Messages"] = dataset_.apply(self.kn_format_message, axis = 1)

        # JSON to str
        dataset_["text"] = self.tokenizer.apply_chat_template(dataset_["Messages"].values.tolist(), tokenize = False)
        dataset_ = Dataset.from_pandas(dataset_)

        return dataset_
