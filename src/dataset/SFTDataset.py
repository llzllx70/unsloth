from src.dataset.BaseDataset import *

class SFTDataset(BaseDataset):
    
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer, 
            flag="sft", 
            origin_dataset_file="data/sft_data.xlsx"
        )

    def add_row_dataset(self, dataset_):

        def f(e):

            if not e["query"] or not e["out"]:
                return None

            return (
                {
                    "query": e["query"],
                    "out": e["out"]
                }
            )

        dataset_2 = dataset_.map(f, remove_columns=dataset_.column_names)
        dataset_filtered = dataset_2.filter(lambda x: x is not None)

        return self.split(dataset_filtered, test_size=0.1)

    def build_dataset(self):

        tr2, te2 = self.add_row_dataset(dataset_=self.origin_dataset)

        self.save([tr2], self.train_file)
        self.save([te2], self.test_file)

    def format(self, x):
        """
        知识+格式训练语料
        """
        query = x["query"]
        out = x["out"]
        
        return [
            {"role": "system", "content": sft_system_prompt},
            {"role": "user", "content": query},
            {"role": "assistant", "content": out},
        ]

    def prepare_dataset(self, dataset_):

        dataset_ = dataset_.to_pandas()[
            ["query", "out"]
        ]

        # pandas to JSON
        dataset_["Messages"] = dataset_.apply(self.format, axis = 1)

        # 对应 SFTTrainer::do_train() -> dataset_text_field = "text",
        dataset_["text"] = self.tokenizer.apply_chat_template(
            dataset_["Messages"].values.tolist(), 
            tokenize = False
        )

        dataset_ = Dataset.from_pandas(dataset_)

        return dataset_
