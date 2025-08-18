
from src.dataset.BaseDataset import *

class PretrainDataset(BaseDataset):
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, flag="pretrain")

    def formatting_prompts_func(self, examples):

        return { "text" : [example + self.tokenizer.eos_token for example in examples["text"]] }

    def build_dataset(self):

        breakpoint()

        ds = load_dataset("roneneldan/TinyStories", split = "train[:25]")

        ds = ds.map(self.formatting_prompts_func, batched = True,)

        for row in ds[:5]["text"]:
            print("=========================")
            print(row)

        tr, te = self.split(dataset_=ds)

        self.save([tr], self.train_file)
        self.save([te], self.test_file)
