
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import argparse

from src.constant.Config import *

parser = argparse.ArgumentParser(description="示例：添加命令行参数")
parser.add_argument("--task", type=str, required=False, help="test")
parser.add_argument("--model", type=str, required=False, help="flag")
parser.add_argument("--step", type=int, required=False, help="flag")
args = parser.parse_args()


class MyMerge:
    
    def __init__(self):

        if args.task == 'pretrain':
            self.base_model = pretrain_base_model
            self.saved_lora = pretrain_saved_lora
            self.merged_model = pretrain_merged_model

        elif args.task == 'sft':
            self.base_model = pretrain_merged_model
            self.saved_lora = sft_saved_lora
            self.merged_model = sft_merged_model

        elif args.task == 'grpo':
            self.base_model = sft_merged_model
            self.saved_lora = grpo_saved_lora
            self.merged_model = grpo_merged_model

    def do_merge(self):

        # 第一步：加载 base 模型（注意路径要匹配你训练时使用的 base）
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # 第二步：加载 LoRA adapter
        lora_model = PeftModel.from_pretrained(
            base_model,
            self.saved_lora,
        )

        # 第三步：合并 LoRA adapter 到 base model
        merged_model = lora_model.merge_and_unload()

        # 第四步：保存合并后的模型
        merged_model.save_pretrained(self.merged_model)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)  # 或原路径
        tokenizer.save_pretrained(self.merged_model)


if __name__ == "__main__":
    merger = MyMerge()
    merger.do_merge()
