
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset

class BaseTrainer:
    
    def format_dataset(self):

        # breakpoint()
        # dataset_ = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
        dataset_ = load_dataset("json", data_files="data/sft.jsonl", split="train")

        dataset_ = dataset_.to_pandas()[
            ["expected_answer", "problem", "generated_solution"]
        ]

        # Try converting to number - if not, replace with NaN
        is_number = pd.to_numeric(pd.Series(dataset_["expected_answer"]), errors = "coerce").notnull()
        # Select only numbers
        dataset_ = dataset_.iloc[np.where(is_number)[0]]

        # pandas to JSON
        dataset_["Messages"] = dataset_.apply(self.format_dataset, axis = 1)

        # truncate to max_seq_length
        dataset_["N"] = dataset_["Messages"].apply(lambda x: len(self.tokenizer.apply_chat_template(x)))
        dataset_ = dataset_.loc[dataset_["N"] <= self.max_seq_length/2].copy()

        # JSON to str
        dataset_["text"] = self.tokenizer.apply_chat_template(dataset_["Messages"].values.tolist(), tokenize = False)
        dataset_ = Dataset.from_pandas(dataset_)

        return dataset_

    
    def format_print(self, query, text, output, use_lora):
        
        # print(f'\n=================lora:{use_lora}==========={query}==========')
        print(f'\n----------lora: {use_lora}-------text--------------------------')
        print(f'{text}')
        print(f'-----------------output------------------------')
        print(f'{output}')
        # print(f'==================End of output====================\n')
        
