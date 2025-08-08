
from src.constant.Config import *
from src.constant.Funs import set_tokenizer_chat_template
from src.dataset.SFTDataset import SFTDataset

from transformers import AutoModelForCausalLM, AutoTokenizer

class MyInfer:
    
    def __init__(self):

        self.model = AutoModelForCausalLM.from_pretrained(sft_merged_model, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(sft_merged_model)

        self.sft_dataset = SFTDataset(self.tokenizer)

    def do_infer(self):

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
    

if __name__ == '__main__':
    
    infer = MyInfer()
    infer.do_infer()
