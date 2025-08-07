
from unsloth import FastLanguageModel
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel

from src.prompt.MyPrompt import *
from src.reward.MyReward import MyReward
from src.trainer.BaseTrainer import BaseTrainer
from src.dataset.GRPODataset import GRPODataset

import argparse

parser = argparse.ArgumentParser(description="示例：添加命令行参数")
parser.add_argument("--task", type=str, required=False, help="test")
parser.add_argument("--model", type=str, required=False, help="flag")
parser.add_argument("--step", type=int, required=False, help="flag")
args = parser.parse_args()


class MyGRPOTrainer(BaseTrainer):
    
    def __init__(self):

        self.sft_saved_lora = f"saved/sft/{args.model}"
        self.saved_lora = f"saved/grpo/{args.model}"
        
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

        self.add_lora1()

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

        self.myreward = MyReward(self.tokenizer)
        self.grpo_dataset = GRPODataset(self.tokenizer)

    def add_lora1(self):

        self.model = PeftModel.from_pretrained(self.model, self.sft_saved_lora)

        # 冻结 lora1 参数
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = False

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
            reward_funcs = self.myreward.build_reward(),
            args = training_args,
            train_dataset = self.grpo_dataset.train_dataset,

            # For optional training + evaluation
            # train_dataset = new_dataset["train"],
            # eval_dataset = new_dataset["test"],
        )
        # self.test()

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


if __name__ == '__main__':
    
    trainer = MyGRPOTrainer()

    if args.task == 'train':
        trainer.do_train()

    if args.task == 'infer':
        trainer.test(use_lora=False)
        trainer.test(use_lora=True)
