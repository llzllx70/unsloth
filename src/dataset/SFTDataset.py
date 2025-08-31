from src.dataset.BaseDataset import *

class SFTDataset(BaseDataset):
    
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer, 
            flag="sft", 
            origin_dataset_file="data/sft_data.xlsx"
        )

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
        reasoning = x["reasoning"].strip()
        
        final_prompt = (
            f'{reasoning_start}{reasoning}{reasoning_end}'
            '\n'
            f'{solution_start}{expected_answer}{solution_end}'
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": final_prompt},
        ]

    def add_whole_row_dataset(self, dataset_):

        def f(e):
            problem = f'浙江省2024年本科{e["专业"]}录取情况'

            return {
                "problem": problem,
                "reasoning": f'好的，针对{problem}，我将从{self.origin_columes}这些方面为您提供相关信息。',
                "expected_answer": self.row_info(prefix=problem, e=e)
            }

        dataset_1 = dataset_.map(f, remove_columns=dataset_.column_names)

        return self.split(dataset_1)

    def add_one_dimension_dataset(self, dataset_):

        def f(e):

            if not e["solution"] or not e["query"] or not e["reasoning"]:
                return None

            return (
                {
                    "problem": e["query"],
                    "reasoning": e["reasoning"],
                    "expected_answer": e["solution"]
                }
            )

        dataset_2 = dataset_.map(f, remove_columns=dataset_.column_names)
        dataset_filtered = dataset_2.filter(lambda x: x is not None)

        return self.split(dataset_filtered, test_size=0.1)

    def build_dataset(self):

        tr2, te2 = self.add_one_dimension_dataset(dataset_=self.origin_dataset)

        self.save([tr2], self.train_file)
        self.save([te2], self.test_file)

    def prepare_dataset(self, dataset_):

        dataset_ = dataset_.to_pandas()[
            ["problem", "reasoning", "expected_answer"]
        ]

        # pandas to JSON
        dataset_["Messages"] = dataset_.apply(self.kn_format_message, axis = 1)

        # JSON to str
        dataset_["text"] = self.tokenizer.apply_chat_template(
            dataset_["Messages"].values.tolist(), 
            tokenize = False
        )

        dataset_ = Dataset.from_pandas(dataset_)

        return dataset_
