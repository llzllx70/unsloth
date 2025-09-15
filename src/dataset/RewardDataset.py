from src.dataset.BaseDataset import *

class RewardDataset(BaseDataset):

    """
    1. 奖励模型的训练数据集来自于GRPO训练过程中的生成数据, 此处只是处理数据保存和加载
    2. 本质上是SFT数据集

    数据格式为：
    {
        "problem": str,
        "reasoning": str,
        "expected_answer": str
    }
    """
    
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer, 
            flag="reward", 
            origin_dataset_file="data/sft_data.xlsx"
        )

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
