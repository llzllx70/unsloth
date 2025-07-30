
from unsloth import FastLanguageModel
import numpy as np
import random
from vllm import SamplingParams
from trl import SFTConfig, SFTTrainer
import os
import json

from datasets import load_dataset, Dataset
import pandas as pd
from transformers import TextStreamer

from MyPrompt import *
from MyReward import MyReward
from BaseTrainer import BaseTrainer

import argparse

parser = argparse.ArgumentParser(description="示例：添加命令行参数")
parser.add_argument("--task", type=str, required=False, help="test")
parser.add_argument("--model", type=str, required=False, help="flag")
parser.add_argument("--step", type=int, required=False, help="flag")
args = parser.parse_args()


class MySFTTrainer(BaseTrainer):
    
    def __init__(self):

        self.saved_lora = f"sft_saved_lora_{args.model}"
        
        self.max_seq_length = 2048 # Can increase for longer reasoning traces
        self.lora_rank = 32 # Larger rank = smarter, but slower

        self.maximum_length = 201
        self.max_prompt_length = self.maximum_length + 1 # + 1 just in case!
        self.max_completion_length = self.max_seq_length - self.max_prompt_length

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = f"models/{args.model}",
            max_seq_length = self.max_seq_length,
            load_in_4bit = False, # False for LoRA 16bit
            fast_inference = True, # Enable vLLM fast inference
            max_lora_rank = self.lora_rank,
            gpu_memory_utilization = 0.7, # Reduce if out of memory
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = self.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha = self.lora_rank*2, # *2 speeds up training
            use_gradient_checkpointing = "unsloth", # Reduces memory usage
            random_state = 3407,
        )

        self.set_tokenizer_chat_template()

        self.infer_sampling_params = SamplingParams(
            temperature = 0.1,
            top_p = 0.95,
            top_k = -1,
            max_tokens = 1024,
        )

        self.myreward = MyReward(self.tokenizer)

        self.train_file = "data/train_sft.jsonl"
        self.test_file = "data/test_sft.jsonl"

        self.train_dataset = self.my_load_dataset(self.train_file)
        self.test_dataset = self.my_load_dataset(self.test_file)

    def kn_message(self, x):
        
        """
        知识训练语料
        """
        expected_answer = x["expected_answer"]
        problem = x["problem"]

        return [
            {"role" : "system",    "content" : sft_system_prompt},
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : expected_answer},
        ]

    def message(self, x):

        expected_answer = x["expected_answer"]
        problem = x["problem"]

        # Remove generated <think> and </think>
        thoughts = x["generated_solution"]
        thoughts = thoughts.replace("<think>", "").replace("</think>", "")

        # Strip newlines on left and right
        thoughts = thoughts.strip()
        # Add our custom formatting

        final_prompt = \
            reasoning_start + thoughts + reasoning_end + \
            solution_start + expected_answer + solution_end

        return [
            {"role" : "system",    "content" : system_prompt},
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : final_prompt},
        ]

    def add_whole_row_dataset(self, dataset_):

        def f(e):
            problem = f'浙江省2024年本科{e["专业"]}录取情况'

            expected_answer = (
                f"浙江省2024年本科{e['专业']}专业的录取计划数为{e['计划数']}人，"
                f"录取数为{e['录取数']}人，省控线为{e['省控线']}分。"
                f"最高分为{e['最高分']}分，最低分为{e['最低分']}分，"
                f"平均分为{e['平均分']}分，最低位次号为{e['最低位次号']}。"
            )

            return {
                "expected_answer": expected_answer,
                "problem": problem,
                "generated_solution": f'好的，针对{problem}，我将从{list(dict(e).keys())}这些方面为您提供相关信息。',
            }

        dataset_1 = dataset_.map(f)

        return dataset_1

    def add_one_dimension_dataset(self, dataset_):

        def f(e):

            r = random.sample(fields, 2)
            train = r[0]
            test = r[1]

            prefix = f'浙江省2024年本科{e["专业"]}专业的{train}'
            dataset_2.append(
                {
                    "expected_answer": f'{prefix}是{e[train]}',
                    "problem": f'{prefix}是多少？',
                    "generated_solution": f'{prefix}是{e[train]}'
                }
            )

            prefix = f'浙江省2024年本科{e["专业"]}专业的{test}'
            with open(self.test_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {
                        "expected_answer": f'{prefix}是{e[test]}',
                        "problem": f'{prefix}是多少？',
                        "generated_solution": f'{prefix}是{e[test]}'
                    },
                    ensure_ascii=False
                ) + "\n")

        fields = ["计划数", "录取数", "省控线", "最高分", "最低分", "平均分", "最低位次号"]

        dataset_2 = []

        dataset_.map(f)

        return dataset_2

    def format_dataset(self):

        dataset_ = load_dataset("json", data_files="data/浙江省2024年本科录取情况.jsonl", split="train")

        dataset_1 = self.add_whole_row_dataset(dataset_=dataset_)
        dataset_2 = self.add_one_dimension_dataset(dataset_=dataset_)

        dataset_1 = dataset_1.to_pandas()[
            ["expected_answer", "problem", "generated_solution"]
        ]

        dataset_ = pd.concat([dataset_1, pd.DataFrame(dataset_2)], ignore_index=True)

        dataset_.to_json(self.train_file, orient="records", lines=True, force_ascii=False)

        return dataset_

    def my_load_dataset(self, jsonl_):

        if os.path.exists(jsonl_):
            dataset_ = load_dataset("json", data_files=jsonl_, split="train")
            dataset_ = dataset_.to_pandas()[
                ["expected_answer", "problem", "generated_solution"]
            ]

        else:
            dataset_ = self.format_dataset()

        # pandas to JSON
        dataset_["Messages"] = dataset_.apply(self.kn_message, axis = 1)

        # JSON to str
        dataset_["text"] = self.tokenizer.apply_chat_template(dataset_["Messages"].values.tolist(), tokenize = False)
        dataset_ = Dataset.from_pandas(dataset_)

        return dataset_

    def set_tokenizer_chat_template(self):

        # Replace with out specific template:
        chat_template = chat_template_\
            .replace("'{system_prompt}'", f"'{system_prompt}'")\
            .replace("'{reasoning_start}'", f"'{reasoning_start}'")

        self.tokenizer.chat_template = chat_template

    def do_train(self):
        
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.train_dataset,
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 1, # Use GA to mimic batch size!
                warmup_steps = 5,
                num_train_epochs = args.step, # Set this for 1 full training run.
                learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
                logging_steps = 5,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                report_to = "none", # Use this for WandB etc
            ),
        )

        # self.test()

        trainer.train()

        self.test(use_lora='1')

        self.model.save_lora(self.saved_lora)

    def do_infer(self, e, use_lora=False):

        text = self.tokenizer.apply_chat_template(
            e["Messages"][:2],
            tokenize = False,
            add_generation_prompt = True, # Must add for generation
        )

        lora_request = self.model.load_lora(self.saved_lora) if use_lora else None

        output = self.model.fast_generate(
            text,
            sampling_params = self.infer_sampling_params,
            lora_request = lora_request
        )[0].outputs[0].text

        self.format_print(query='', text=text, output=output, use_lora=use_lora)

    def test(self, use_lora=False):

        for e in self.test_dataset:
            self.do_infer(e, use_lora=use_lora)


if __name__ == '__main__':
    
    trainer = MySFTTrainer()

    if args.task == 'train':
        trainer.do_train()

    if args.task == 'infer':
        # trainer.test(use_lora=False)
        trainer.test(use_lora=True)
