from dataset.BaseDataset import *

class GRPODataset(BaseDataset):
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, flag="grpo")

    def score_judge(self):

        """
        {
            "expected_answer":"浙江省2024年本科医学检验技术专业的最低位次号是183515",
            "problem":"浙江省2024年本科医学检验技术专业的最低位次号是多少？",
            "generated_solution":"好的，针对浙江省2024年本科医学检验技术专业的最低位次号是多少？的问题，可以搜索到如下相关信息浙江省2024年本科医学检验技术专业的录取计划数为43人，录取数为43人，省控线为492分。最高分为541分，最低分为486分，平均分为506分，最低位次号为183515。"
        }

        """

        my_data = [
            {
                "prompt" : [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": q}
                ],
                "answer": a,
                "ref_reason": r
            }
            for q, a, r in MyTrainDataset
        ]


    def build_dataset(self):

        """
        1. 分数判断: “我考了”“分” → “能否”                                                                                
        2. 位次判断: “我排在”“位” → “能不能”                                                                              
        3. 推荐: “推荐什么”“适合我”                                                                                       
        4. 信息查询: “平均分”“最低分”  
        """

        a = self.score_judge()
        b = self.rank_judge()
        c = self.recommend()
        d = self.query_info()

        return concatenate_datasets([a, b, c, d])

    def prepare_dataset(self, dataset_):
        return dataset_
