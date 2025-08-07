
from unsloth import FastLanguageModel
from vllm import SamplingParams
from trl import SFTConfig, SFTTrainer

from prompt.MyPrompt import *
from reward.MyReward import MyReward
from trainer.BaseTrainer import BaseTrainer
from dataset.SFTDataset import SFTDataset

import argparse

parser = argparse.ArgumentParser(description="示例：添加命令行参数")
parser.add_argument("--task", type=str, required=False, help="test")
parser.add_argument("--model", type=str, required=False, help="flag")
parser.add_argument("--step", type=int, required=False, help="flag")
args = parser.parse_args()


class MySFTTrainer(BaseTrainer):
    
    def __init__(self):

        self.saved_lora = f"saved/sft/{args.model}"
        
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

        self.sft_dataset = SFTDataset(self.tokenizer)


    def do_train(self):
        
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.sft_dataset.train_dataset,
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
        self.model.save_lora(self.saved_lora)

        self.test(use_lora='1')

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

        for e in self.sft_dataset.test_dataset:
            self.do_infer(e, use_lora=use_lora)


if __name__ == '__main__':
    
    trainer = MySFTTrainer()

    if args.task == 'train':
        trainer.do_train()

    if args.task == 'infer':
        # trainer.test(use_lora=False)
        trainer.test(use_lora=True)
