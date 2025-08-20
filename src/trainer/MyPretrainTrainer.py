
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from transformers import TextIteratorStreamer
from threading import Thread
import textwrap

from src.prompt.MyPrompt import *
from src.trainer.BaseTrainer import BaseTrainer
from src.dataset.PretrainDataset import PretrainDataset
from src.constant.Config import *


import argparse

parser = argparse.ArgumentParser(description="示例：添加命令行参数")
parser.add_argument("--task", type=str, required=False, help="test")
parser.add_argument("--model", type=str, required=False, help="flag")
parser.add_argument("--step", type=int, required=False, help="flag")
args = parser.parse_args()


class MyPretrainTrainer(BaseTrainer):

    def __init__(self):

        self.max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
        self.dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        self.load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

        self.max_print_width = 100

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = f"models/{args.model}", # "unsloth/mistral-7b" for 16bit loading
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj", 
                "embed_tokens", "lm_head", # Add for continual pretraining
            ],     
            lora_alpha = 32,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = True,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

        self.pretrain_dataset = PretrainDataset(self.tokenizer)
        self.text_streamer = TextIteratorStreamer(self.tokenizer)

    def do_train(self):

        trainer = UnslothTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.pretrain_dataset.train_dataset,
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            dataset_num_proc = 8,

            args = UnslothTrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 8,

                warmup_ratio = 0.1,
                num_train_epochs = args.step,

                learning_rate = 5e-5,
                embedding_learning_rate = 5e-6,

                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.00,
                lr_scheduler_type = "cosine",
                seed = 3407,
                output_dir = "outputs",
                report_to = "none", # Use this for WandB etc
            ),
        )

        trainer_stats = trainer.train()
        self.save()

    def save(self):

        print(type(self.model))

        self.model.save_pretrained(pretrain_saved_lora)
        self.tokenizer.save_pretrained(pretrain_saved_lora)

    def do_infer(self, e):
        
        inputs = self.tokenizer(
        [
            "Once upon a time, in a galaxy, far far away,"
        ]*1, return_tensors = "pt").to("cuda")

        generation_kwargs = dict(
            inputs,
            streamer = self.text_streamer,
            max_new_tokens = 256,
            use_cache = True,
        )
        thread = Thread(target = self.model.generate, kwargs = generation_kwargs)
        thread.start()

        length = 0
        for j, new_text in enumerate(self.text_streamer):
            if j == 0:
                wrapped_text = textwrap.wrap(new_text, width = self.max_print_width)
                length = len(wrapped_text[-1])
                wrapped_text = "\n".join(wrapped_text)
                print(wrapped_text, end = "")
            else:
                length += len(new_text)
                if length >= self.max_print_width:
                    length = 0
                    print()
                print(new_text, end = "")
            pass
        pass

    def test(self):

        for e in self.pretrain_dataset.test_dataset:
            self.do_infer(e)
            break


if __name__ == '__main__':
    
    trainer = MyPretrainTrainer()

    if args.task == 'train':
        trainer.do_train()
