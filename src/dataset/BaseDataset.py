import random
import json
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import os
from transformers import TextStreamer
import shutil

from src.prompt.MyPrompt import *

def year(row):
    return f"{row['年份']}年"

def province(row):
    return f"{row['省份']}省"

def level(row):
    return f"{row['层次']}"

def category(row):
    return f"{row['类别']}"

def major(row):
    return f"{row['专业']}专业"

def min_score(row):
    return f"最低分{row['最低分']}"

def avg_score(row):
    return f"平均分{row['平均分']}"

def max_score(row):
    return f"最高分{row['最高分']}"

def admission_num(row):
    return f"录取数{row['录取数']}"

def control_line(row):
    return f"省控线{row['省控线']}"

def plan_num(row):
    return f"录取计划数{row['计划数']}"

def min_rank(row):
    return f"最低位次号{row['最低位次号']}"


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
            f"{prefix}: {plan_num(e)}人，"
            f"{admission_num(e)}人，{control_line(e)}分。"
            f"{max_score(e)}分， {min_score(e)}分，"
            f"{avg_score(e)}分， {min_rank(e)}。"
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


