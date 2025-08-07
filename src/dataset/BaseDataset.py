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
        self.train_file = f"data/{flag}_train.jsonl"
        self.test_file = f"data/{flag}_test.jsonl"

        self.origin_dataset_ = load_dataset("json", data_files="data/浙江省2024年本科录取情况.jsonl", split="train")

        self.train_dataset = self.loading_dataset(self.train_file)
        self.test_dataset = self.loading_dataset(self.test_file)

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



