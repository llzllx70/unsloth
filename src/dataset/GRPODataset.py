from src.dataset.BaseDataset import *

class GRPODataset(BaseDataset):
    
    def __init__(self, tokenizer):
        super().__init__(
            tokenizer=tokenizer, 
            flag="grpo",
            origin_dataset_file=None
        )

    def score_judge(self, dataset_):

        """
        1. 分数判断: 我考了xx分, 能否报考yy专业？                                                                               
        """

        def q(s_, z_):
            return f'我考了{s_}分，能否报考{z_}专业？'

        def f(e):
            s_ = random.randint(450, 600)
            z_ = e["专业"]

            q_ = q(s_, z_)

            return {
                "prompt" : [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": q_}
                ],
                "task": "score_judge",
                "info": {
                    "score": s_,
                    "detail": e
                }
            }

        d = dataset_.map(f, remove_columns=dataset_.column_names)

        return self.split(d)

    def rank_judge(self, dataset_):

        """
        1. 分数判断: 我考了xx分, 能否报考yy专业？                                                                               
        """

        def q(r_, z_):
            return f'我的位次为{r_}，能否报考{z_}专业？'

        def f(e):
            r_ = random.randint(100000, 200000)
            z_ = e["专业"]

            q_ = q(r_, z_)

            return {
                "prompt" : [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": q_}
                ],
                "task": "rank_judge",
                "info": {
                    "rank": r_,
                    "detail": e
                }
            }

        d = dataset_.map(f, remove_columns=dataset_.column_names)

        return self.split(d)

    def from_sft(sefl):
        
        """
        从sft语料中构建grpo语料
        """

        def f(e):
            return {
                "prompt" : [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": e["problem"]}
                ],
                "task": "F1",
                "reasoning": e["reasoning"],
                "solution": e["expected_answer"]
            }
            
        sft_train = load_dataset('json', data_files='data/sft_train.jsonl', split='train')
        sft_test = load_dataset('json', data_files='data/sft_test.jsonl', split='train')

        tr = sft_train.map(f, remove_columns=sft_train.column_names)
        te = sft_test.map(f, remove_columns=sft_test.column_names)

        return tr, te

    def build_dataset(self):

        """
        1. 分数判断: “我考了”“分” → “能否”                                                                                
        2. 位次判断: “我排在”“位” → “能不能”                                                                              
        3. 推荐: “推荐什么”“适合我”                                                                                       
        4. 信息查询: “平均分”“最低分”  
        """

        # tr1, te1 = self.score_judge(self.origin_dataset_)
        # tr2, te2 = self.rank_judge(self.origin_dataset_)
        # c = self.recommend()
        # d = self.query_info()

        # self.save([tr1, tr2], self.train_file)
        # self.save([te1, te2], self.test_file)

        tr, te = self.from_sft()
        self.save([tr], self.train_file)
        self.save([te], self.test_file)

