
from src.dataset.BaseDataset import *

class PretrainDataset(BaseDataset):
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, flag="pretrain")

    def formatting_prompts_func(self, examples):
        return { "text" : [example + self.tokenizer.eos_token for example in examples["text"]] }

    def trunc(self, e):
        return { "text" : " ".join(e["text"].split()[:10]) }

    def build_dataset(self):

        ds = load_dataset("roneneldan/TinyStories", split = "train[:25]")
        tr = ds.map(self.formatting_prompts_func, batched = True,)
        te = tr.map(self.trunc)

        self.save([tr], self.train_file)
        self.save([te], self.test_file)
