from MyPrompt import *
import random
import json
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import os
from transformers import TextStreamer

class BaseDataset:
    
    def __init__(self, tokenizer, flag):

        self.tokenizer = tokenizer
        self.train_file = f"data/{flag}_train.jsonl"
        self.test_file = f"data/{flag}_test.jsonl"

        self.origin_dataset_ = load_dataset("json", data_files="data/浙江省2024年本科录取情况.jsonl", split="train")

        self.train_dataset = self.loading_dataset(self.train_file)
        self.test_dataset = self.loading_dataset(self.test_file)

    def set_tokenizer_chat_template(self):

        chat_template = chat_template_\
            .replace("'{system_prompt}'", f"'{system_prompt}'")\
            .replace("'{reasoning_start}'", f"'{reasoning_start}'")

        self.tokenizer.chat_template = chat_template

    def loading_dataset(self, jsonl_):

        if not os.path.exists(jsonl_):
            self.build_dataset()

        dataset_ = load_dataset("json", data_dir=jsonl_, split="train")
        return self.prepare_dataset(dataset_)

