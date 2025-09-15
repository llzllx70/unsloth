from src.dataset.BaseDataset import *
from src.dataset.SFTDataset import SFTDataset
from src.nl2sql.LLMApi import LLMApi

class RewardDataset(BaseDataset):

    """
    1. 奖励模型的训练数据集来自于GRPO训练过程中的生成数据, 此处只是处理数据保存和加载
    2. 本质上是SFT数据集

    数据格式为：
    {
        "query": str,
        "reasoning": str,
        "solution": str
        "score": float
    }
    """
    
    def __init__(self, tokenizer):

        self.llm_api = LLMApi()

        super().__init__(
            tokenizer=tokenizer, 
            flag="reward", 
            origin_dataset_file="data/sft_data.xlsx"

        )

    def reward_score(self, md, q, r, s):

        return self.llm_api.reward_score(
            md=md,
            query=q,
            reasoning=r,
            solution=s
        )

    def add_row_dataset(self, dataset_):

        def f(e):

            md, q, r, s = e["result"], e["query"], e["reasoning"], e["solution"]
            if not md or not q or not r or not s: return None

            score = self.reward_score(md, q, r, s)
            if not score: return None

            return (
                {
                    "query": q,
                    "reasoning": r,
                    "solution": s,
                    "score": score
                }
            )

        dataset_2 = dataset_.map(f, remove_columns=dataset_.column_names)
        dataset_filtered = dataset_2.filter(lambda x: x is not None)

        return self.split(dataset_filtered, test_size=0.1)

    def add_score(self, dataset_, file_):

        """在SFT的基础上添加score字段作为奖励信号"""

        dataset_ = dataset_.to_pandas()[
            ["query", "reasoning", "solution"]
        ]

        dataset_["score"] = dataset_.apple(self.reward_score, axis=1)

        self.save([Dataset.from_pandas(dataset_)], file_)

    def build_dataset(self):

        tr2, te2 = self.add_row_dataset(dataset_=self.origin_dataset)

        self.save([tr2], self.train_file)
        self.save([te2], self.test_file)

    def kn_format_message(self, x):
        """
        知识+格式训练语料
        """
        solution = x["solution"]
        query = x["query"]
        reasoning = x["reasoning"].strip()
        
        final_prompt = (
            f'{reasoning_start}{reasoning}{reasoning_end}'
            '\n'
            f'{solution_start}{solution}{solution_end}'
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
            {"role": "assistant", "content": final_prompt},
        ]

    def prepare_dataset(self, dataset_):

        dataset_ = dataset_.to_pandas()[
            ["query", "reasoning", "solution"]
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


if __name__ == "__main__":
    
    reward_dataset = RewardDataset(tokenizer=None)
    reward_dataset.build_dataset()
