
from src.dataset.BaseDataset import *

class PretrainDataset(BaseDataset):
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, flag="pretrain")

    def add_whole_row_dataset(self, dataset_):

        def f(e):
            prefix = f'浙江省2024年本科{e["专业"]}录取情况'
            return {
                "text": self.row_info(prefix, e),
                "prefix": prefix
            }

        return dataset_.map(f, remove_columns=dataset_.column_names)

    def build_dataset(self):

        tr = self.add_whole_row_dataset(dataset_=self.origin_dataset_)

        self.save([tr], self.train_file)
        self.save([tr], self.test_file)
