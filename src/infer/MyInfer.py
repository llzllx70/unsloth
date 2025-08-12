
from src.constant.Config import *
from src.constant.Funs import set_tokenizer_chat_template
from src.dataset.SFTDataset import SFTDataset
from src.dataset.GRPODataset import GRPODataset

from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="示例：添加命令行参数")
parser.add_argument("--task", type=str, required=False, help="test")
parser.add_argument("--model", type=str, required=False, help="flag")
parser.add_argument("--step", type=int, required=False, help="flag")
args = parser.parse_args()


class MyInfer:
    
    def __init__(self, task='sft', model='Qwen3-4B-Base'):

        self.task = task

        merged_model = f'merged/{self.task}/{model}'
        self.model = AutoModelForCausalLM.from_pretrained(merged_model, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(merged_model)

        self.sft_dataset = SFTDataset(self.tokenizer)
        self.grpo_dataset = GRPODataset(self.tokenizer)

    def do_sft_infer(self):

        ret = []

        set_tokenizer_chat_template(self.tokenizer)

        for e in self.sft_dataset.test_dataset:
            text = self.tokenizer.apply_chat_template(
                e["Messages"][:2],
                tokenize=False,
                add_generation_prompt=True,  # Must add for generation
            )

            output = self.model.generate(
                **self.tokenizer(text, return_tensors="pt").to("cuda"),
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
            )

            output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            print(f"Input: {text}")
            print(f"Output: {output_text}\n")    

            ret.append({
                f'{self.task}_text': text,
                f'{self.task}_output': output_text 
            })

        return ret

    def do_grpo_infer(self):
        
        ret = []

        set_tokenizer_chat_template(self.tokenizer)

        for item in self.grpo_dataset.test_dataset:
            
            text = self.tokenizer.apply_chat_template(
                item['prompt'],
                tokenize = False, 
                add_generation_prompt = True
            )

            output = self.model.generate(
                **self.tokenizer(text, return_tensors="pt").to("cuda"),
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
            )

            output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            print(f"Input: {text}")
            print(f"Output: {output_text}\n")    

            ret.append({
                f'{self.task}_text': text,
                f'{self.task}_output': output_text 
            })

        return ret

def compare_infer():

    ret1 = MyInfer(task='sft', model=args.model).do_grpo_infer()
    ret2 = MyInfer(task='grpo', model=args.model).do_grpo_infer()

    ret = []

    for a, b in zip(ret1, ret2):
        ret.append({**a, **b})     

    df = pd.DataFrame(ret)
    df.to_excel('compare_infer.xlsx', index=False)
    

if __name__ == '__main__':
    compare_infer()
