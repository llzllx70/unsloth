
from src.dataset.BaseDataset import *

class PretrainDataset(BaseDataset):
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, flag="pretrain")

    def add_whole_row_dataset(self, dataset_):

        def f(e):
            return {
                "text": self.row_info(e),
                "prefix": f'{e["年份"]}年{e["省份"]}{e["层次"]}{e["类别"]}{e["专业"]}录取情况：'
            }

        return dataset_.map(f, remove_columns=dataset_.column_names)

    def build_dataset(self):

        tr = self.add_whole_row_dataset(dataset_=self.origin_dataset)

        self.save([tr], self.train_file)
        self.save([tr], self.test_file)
