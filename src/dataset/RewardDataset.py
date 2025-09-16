from src.dataset.BaseDataset import *
from src.nl2sql.LLMApi import LLMApi
from src.infer.MyInfer import SFTInfer
from src.common.MyRe import MyRe

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

        self.sft_infer = SFTInfer()
        self.re = MyRe()
        self.llm_api = LLMApi()

        super().__init__(
            tokenizer=tokenizer, 
            flag="reward", 
            origin_dataset_file="data/sft_data.xlsx"
        )

    def do_infer_reasoning_solution(self, q):

        try:
            output_text = self.sft_infer.default_infer(q)
            reasoning, solution = self.re.extract_reasoning_solution(output_text)

            return reasoning, solution

        except Exception as ex:
            print(f"Error: {ex}")
            return None, None

    def add_row_dataset(self, dataset_):

        def f(e):

            q, sql, md, r, s = e["query"], e["sql"], e["result"], e["reasoning"], e["solution"]
            if not q or not md or not r or not s: return None

            ret = {
                "query": q,
                "sql": sql,
                "result": md,
                "reasoning": r,
                "solution": s 
            }

            for i in range(3):

                ri, si = self.do_infer_reasoning_solution(q) 
                score = self.llm_api.reward_score(
                    reasoning=r,
                    solution=s,
                    reasoningi=ri,
                    solutioni=si
                )

                ret.update({
                    f"reasoning{i}": ri,
                    f"solution{i}": si,
                    f"score{i}": score
                })


        # dataset_2 = dataset_.map(f, remove_columns=dataset_.column_names)
        dataset_2 = dataset_.select(range(2)).map(f, remove_columns=dataset_.column_names)
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
