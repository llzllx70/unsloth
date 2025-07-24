
from unsloth import FastLanguageModel
import numpy as np
from vllm import SamplingParams
from trl import SFTConfig, SFTTrainer

from datasets import load_dataset, Dataset
import pandas as pd
from transformers import TextStreamer

from MyPrompt import *
from MyReward import MyReward

import argparse

parser = argparse.ArgumentParser(description="示例：添加命令行参数")
parser.add_argument("--task", type=str, required=False, help="test")
parser.add_argument("--model", type=str, required=False, help="flag")
parser.add_argument("--step", type=int, required=False, help="flag")
args = parser.parse_args()


class MySFTTrainer:
    
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

        self.dataset = self.build_dataset()

    def format_dataset(self, x):

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

    def build_dataset(self):

        dataset_ = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
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
            train_dataset = self.dataset,
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 1, # Use GA to mimic batch size!
                warmup_steps = 5,
                num_train_epochs = 2, # Set this for 1 full training run.
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

    def test(self, use_lora=False):

        breakpoint()
        text = self.tokenizer.apply_chat_template(
            self.dataset[0]["Messages"][:2],
            tokenize = False,
            add_generation_prompt = True, # Must add for generation
        )

        output = self.model.generate(
            **self.tokenizer(text, return_tensors = "pt").to("cuda"),
            temperature = 0,
            max_new_tokens = 1024,
            streamer = TextStreamer(self.tokenizer, skip_prompt = False),
        )

        self.format_print(query='', text=text, output=output[0].text, use_lora=use_lora)

    def format_print(self, query, text, output, use_lora):
        
        print(f'\n=================lora:{use_lora}==========={query}==========')
        print(f'-----------------text--------------------------')
        print(f'{text}')
        print(f'\n-----------------output------------------------')
        print(f'{output}')
        print(f'==================End of output====================\n')
        

if __name__ == '__main__':
    
    trainer = MySFTTrainer()

    if args.task == 'train':
        trainer.do_train()

    if args.task == 'infer':
        trainer.test(use_lora=False)
