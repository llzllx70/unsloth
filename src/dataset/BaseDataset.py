import random
import json
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import os
from transformers import TextStreamer

from src.prompt.MyPrompt import *

class BaseDataset:
    
    def __init__(self, tokenizer, flag, origin_dataset_file):

        self.tokenizer = tokenizer

        self.origin_dataset_file = origin_dataset_file

        self.train_file = f"data/{flag}_train.jsonl"
        self.test_file = f"data/{flag}_test.jsonl"

        self.train_dataset = self.loading_dataset(self.train_file)
        self.test_dataset = self.loading_dataset(self.test_file)

    @property
    def origin_df(self):
        return pd.read_excel(self.origin_dataset_file)
    
    @property
    def origin_dataset(self):
        return Dataset.from_pandas(self.origin_df)

    @property
    def origin_columes(self):
        return self.origin_dataset.column_names
        
    def row_info(self, prefix, e):

        return (
            f"{prefix}: 录取计划数为{e['计划数']}人，"
            f"录取数为{e['录取数']}人，省控线为{e['省控线']}分。"
            f"最高分为{e['最高分']}分，最低分为{e['最低分']}分，"
            f"平均分为{e['平均分']}分，最低位次号为{e['最低位次号']}。"
        )

    def loading_dataset(self, jsonl_):

        if not os.path.exists(jsonl_):
            self.build_dataset()

        dataset_ = load_dataset("json", data_files=jsonl_, split="train")
        return self.prepare_dataset(dataset_)

    def split(self, dataset_, test_size=0.2, seed=42):

        s = dataset_.train_test_split(test_size=test_size, seed=seed)
        return s['train'], s['test']

    def save(self, l_dataset_, jsonl_):
        
        dataset_ = concatenate_datasets(l_dataset_).shuffle(seed=42)
        dataset_.to_json(jsonl_, orient="records", lines=True, force_ascii=False)

    def prepare_dataset(self, dataset_):
        return dataset_


