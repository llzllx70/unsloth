
from src.dataset.BaseDataset import *

class PretrainDataset(BaseDataset):
    
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer, 
            flag="pretrain",
            origin_dataset_file="data/2022_23_24年浙江树人学院各省份录取情况.xlsx"
        )

    def add_whole_row_dataset(self, dataset_):

        def f(e):
            prefix = f'浙江树人学院{e["年份"]}年{e["省份"]}{e["层次"]}{e["类别"]}{e["专业"]}录取情况：'
            return {
                "text": self.row_info(prefix, e),
                "prefix": prefix
            }

        return dataset_.map(f, remove_columns=dataset_.column_names)

    def build_dataset(self):

        tr = self.add_whole_row_dataset(dataset_=self.origin_dataset)

        self.save([tr], self.train_file)
        self.save([tr], self.test_file)
