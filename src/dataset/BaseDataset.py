import random
import json
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import os
from transformers import TextStreamer

from src.prompt.MyPrompt import *

class BaseDataset:
    
    def __init__(self, tokenizer, flag):

        self.tokenizer = tokenizer

        self.origin_dataset_file = "data/2022_23_24年浙江树人学院各省份录取情况.xlsx"

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
        
    def row_info(self, e):
        return '，'.join(f"{k}为{v}" for k, v in e.items())

    def loading_dataset(self, jsonl_):

        if not os.path.exists(jsonl_):
            self.build_dataset()

        dataset_ = load_dataset("json", data_files=jsonl_, split="train")
        return self.prepare_dataset(dataset_)

    def split(self, dataset_, test_size=0.2, seed=42):

        s = dataset_.train_test_split(test_size=test_size, seed=seed)
        return s['train'], s['test']

    def save(self, l_dataset_, jsonl_):
        
        dataset_ = concatenate_datasets(l_dataset_)
        dataset_.to_json(jsonl_, orient="records", lines=True, force_ascii=False)

    def prepare_dataset(self, dataset_):
        return dataset_


