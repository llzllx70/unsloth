
from unsloth import FastLanguageModel
import torch
import re
import numpy as np
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer

from datasets import load_dataset, Dataset
import pandas as pd

from MyPrompt import *

import argparse

parser = argparse.ArgumentParser(description="示例：添加命令行参数")
parser.add_argument("--task", type=str, required=False, help="test")
parser.add_argument("--model", type=str, required=False, help="flag")
parser.add_argument("--step", type=int, required=False, help="flag")
args = parser.parse_args()


class MyGRPOTrainer:
    
    def __init__(self):

        self.saved_lora = f"grpo_saved_lora_{args.model}"
        
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

        self.match_format = self.build_match_format()

        self.set_tokenizer_chat_template()

        self.vllm_sampling_params = SamplingParams(
            min_p = 0.1,
            top_p = 1.0,
            top_k = -1,
            seed = 3407,
            stop = [self.tokenizer.eos_token],
            include_stop_str_in_output = True,
        )
        
        self.infer_sampling_params = SamplingParams(
            temperature = 0.1,
            top_p = 0.95,
            top_k = -1,
            max_tokens = 1024,
        )

    @property
    def dataset(self):

        math = False
        if math:
            dataset_ = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split = "train")

            dataset_ = dataset_.map(lambda x: {
                "prompt" : [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": x["prompt"]},
                ],
                "answer": x["solution"],
                "xx": "yy"
            }) 

        add = True
        if add:
            my_data = [
                {
                    "prompt" : [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": q}
                    ],
                    "answer": a,
                    "ref_reason": r
                }
                for q, a, r in MyTrainDataset
            ]

            my_dataset = Dataset.from_pandas(pd.DataFrame(my_data))

        if math and add:
            from datasets import concatenate_datasets
            dataset_ = concatenate_datasets([dataset_, my_dataset])

        else:
            dataset_ = my_dataset

        tokenized = dataset_.map(
            lambda x: {
                "tokens": self.tokenizer.apply_chat_template(
                    x["prompt"], 
                    add_generation_prompt = True, 
                    tokenize = True
                )
            },
            batched = True,
        )
        print(self.tokenizer.decode(tokenized[0]["tokens"]))

        tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})

        a = dataset_.select(np.where(np.array(tokenized["L"]) <= self.maximum_length)[0])

        return a

    def set_tokenizer_chat_template(self):

        # Replace with out specific template:
        chat_template = chat_template_\
            .replace("'{system_prompt}'", f"'{system_prompt}'")\
            .replace("'{reasoning_start}'", f"'{reasoning_start}'")

        self.tokenizer.chat_template = chat_template


    def build_match_format(self):
        
        # Add optional EOS token matching
        solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
            "(?:" + re.escape(self.tokenizer.eos_token) + ")?"

        return re.compile(
            rf"{reasoning_end}.*?"\
            rf"{solution_start}(.+?){solution_end_regex}"\
            rf"[\s]{{0,}}$",
            flags = re.MULTILINE | re.DOTALL
        )
        
    def score_print(self, scores, flag):

        print(f'\n=================score:{flag}=====================')
        print(f'{scores}')

    def match_format_exactly(self, completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Match if format is seen exactly!
            if self.match_format.search(response) is not None: 
                score += 3.0
            scores.append(score)

        self.score_print(scores=scores,flag='match_format_exactly')
        return scores

    def F1_reward(self, completions, ref_reason, **kwargs):

        scores = []

        for idx, completion in enumerate(completions):

            pred_reason = completion[0]['content']

            pred_tokens = set(pred_reason.strip().split())
            ref_tokens = set(ref_reason[idx].strip().split())

            overlap = pred_tokens & ref_tokens
            if not overlap:
                return 0.0
            precision = len(overlap) / len(pred_tokens)
            recall = len(overlap) / len(ref_tokens)
            scores.append(2 * precision * recall / (precision + recall))
            
        self.score_print(scores=scores,flag='F1_reward')
        return scores

    def check_answer(self, prompts, completions, answer, **kwargs):

        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := self.match_format.search(r)) is not None else None \
            for r in responses
        ]

        print(extracted_responses)

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0

            if guess is None:
                score = -2.0

            elif guess == true_answer:
                score += 5.0

            elif guess.strip() == true_answer.strip():
                score += 3.5

            elif guess.strip() in ['3', '4', '6']:
                score -= 6

            else:
                score += 1

            scores.append(score)

        self.score_print(scores=scores,flag='check_answer')
        return scores

    def do_train(self):
        
        training_args = GRPOConfig(
            vllm_sampling_params = self.vllm_sampling_params,
            temperature = 1.0,
            # learning_rate = 5e-6,
            learning_rate = 5e-5,
            weight_decay = 0.01,
            warmup_ratio = 0.1,
            lr_scheduler_type = "linear",
            optim = "adamw_8bit",
            logging_steps = 1,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1, # Increase to 4 for smoother training
            num_generations = 4, # Decrease if out of memory
            max_prompt_length = self.max_prompt_length,
            max_completion_length = self.max_completion_length,
            num_train_epochs = args.step, # Set to 1 for a full training run
            max_steps = args.step,
            save_steps = args.step,
            report_to = "none", # Can use Weights & Biases
            output_dir = "outputs",

            # For optional training + evaluation
            # fp16_full_eval = True,
            # per_device_eval_batch_size = 4,
            # eval_accumulation_steps = 1,
            # eval_strategy = "steps",
            # eval_steps = 1,
        )

        trainer = GRPOTrainer(
            model = self.model,
            processing_class = self.tokenizer,
            reward_funcs = [
                self.match_format_exactly,
                # self.F1_reward,
                self.check_answer
            ],
            args = training_args,
            train_dataset = self.dataset,

            # For optional training + evaluation
            # train_dataset = new_dataset["train"],
            # eval_dataset = new_dataset["test"],
        )
        self.test()

        trainer.train()

        self.test(use_lora='1')

        self.model.save_lora(self.saved_lora)

    def do_infer(self, query, use_lora=False):

        text = self.tokenizer.apply_chat_template([
            {"role" : "system", "content" : system_prompt},
            {"role" : "user", "content" : query},
        ], tokenize = False, add_generation_prompt = False)

        if use_lora == '1':
            output = self.model.fast_generate(
                text,
                sampling_params = self.infer_sampling_params,
            )[0].outputs[0].text

        else:
            lora_request = self.model.load_lora(self.saved_lora) if use_lora else None

            output = self.model.fast_generate(
                text,
                sampling_params = self.infer_sampling_params,
                lora_request = lora_request
            )[0].outputs[0].text

        self.format_print(query, text, output, use_lora)

    def test(self, use_lora=False):

        for q, a, r in MyTrainDataset:
            self.do_infer(q, use_lora=use_lora)

    def format_print(self, query, text, output, use_lora):
        
        print(f'\n=================lora:{use_lora}==========={query}==========')
        print(f'-----------------text--------------------------')
        print(f'{text}')
        print(f'\n-----------------output------------------------')
        print(f'{output}')
        print(f'==================End of output====================\n')
        

if __name__ == '__main__':
    
    trainer = MyGRPOTrainer()

    if args.task == 'train':
        trainer.do_train()

    if args.task == 'infer':
        trainer.test(use_lora=False)
        trainer.test(use_lora=True)
        trainer.test(use_lora=False)
