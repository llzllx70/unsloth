
import jieba
import json
from src.common.MyRe import MyRe
from src.reward.kw import all_kw
from src.reward.bge import Bge

for w in all_kw:
    jieba.add_word(w)

class MyReward:

    def __init__(self, tokenizer=None):

        if tokenizer is None:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = tokenizer.eos_token

        self.re = MyRe()

        self.stop_words = set(['的', '了', '和', '是', '在', '，', '。', '：', '（', '）', '-', '\n', ' ', '有', '与', '或', '您'])
        self.bge = Bge()

    def score_print(self, origin_scores, scores, flag, infos=None):

        print(f'\n=================score:{flag}=====================')
        print(f'origin_score: {origin_scores} -> {scores}')

        if infos is not None:
            pretty_json = json.dumps(infos, indent=2, ensure_ascii=False)
            print(pretty_json)

    def start_with_reasoning_reward(self, completions, **kwargs):

        scores = []

        for completion in completions:
            response = completion[0]["content"]
            score = 10 if response.strip().startswith("<REASONING>") else 0.0
            scores.append(score)

        self.score_print(scores=scores,flag='0. start_with_reasoning_reward')
        return scores

    def F1_reward(self, completions, query, out, **kwargs):

        origin_scores = []
        truth = out[0]
        truth_tokens = list(jieba.lcut(truth))
        truth_tokens = [t for t in truth_tokens if t not in self.stop_words]

        infos = {
            'truth': truth,
            'pred': []
        }

        for idx, completion in enumerate(completions):
            pred = completion[0]['content']
            pred_tokens = list(jieba.lcut(pred))
            pred_tokens = [t for t in pred_tokens if t not in self.stop_words]

            common = list(set(pred_tokens) & set(truth_tokens))
            common_count = sum(min(pred_tokens.count(t), truth_tokens.count(t)) for t in common)

            if common_count == 0:
                f1 = 0.0
            else:
                precision = common_count / len(pred_tokens)
                recall = common_count / len(truth_tokens)
                f1 = 2 * precision * recall / (precision + recall)

            origin_scores.append(f1)
            infos['pred'].append({
                "pred": pred,
                # "pred_tokens": pred_tokens,
                # "truth_tokens": truth_tokens,
                # "common": common,
                "common_count": common_count,
                # "f1": f1
            })
            
        # if query[0] == '我是浙江考生，选课物化，我分数482，排名192244，被录取的概率有多大？':
        """
        从pretrain的基础上进行训练，不需要过滤， 因为起点比较低
        if max(origin_scores) < 0.7:
            scores = [0.0 for _ in origin_scores]

        else:
            scores = origin_scores
        """

        scores = origin_scores
        
        self.score_print(origin_scores=origin_scores, scores=scores, flag=f'F1 {query[0]}', infos=infos)

        return scores

    def F1_bge(self, completions, query, out, **kwargs):

        origin_scores = []
        truth = out[0]
        truth_tokens = list(jieba.lcut(truth))
        truth_tokens = [t for t in truth_tokens if t not in self.stop_words]

        ss = [completion[0]['content'] for completion in completions]

        infos = {
            'truth': truth,
            'pred': ss
        }

        origin_scores = self.bge.scores(truth, ss)

        """
        从pretrain的基础上进行训练，不需要过滤， 因为起点比较低
        if max(origin_scores) < 0.7:
            scores = [0.0 for _ in origin_scores]

        else:
            scores = origin_scores
        """

        scores = origin_scores
        
        self.score_print(origin_scores=origin_scores, scores=scores, flag=f'F1 {query[0]}', infos=infos)

        return scores

    def check_answer(self, prompts, completions, answer, **kwargs):

        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [self.re.extract_solution(r) for r in responses]

        print(extracted_responses)

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0

            if guess is None:
                score = -2.0

            elif guess == true_answer:
                score += 5.0

            elif guess.strip() == true_answer.strip():
                score += 3.5

            scores.append(score)

        self.score_print(scores=scores,flag='3. check_answer')
        return scores

    def format_score(self, prompts, completions, task, info, **kwargs):

        scores = []

        for completion, info_ in zip(completions, info):

            response = completion[0]["content"]

            reasoning, solution = self.re.extract_reasoning_solution(response)

            score = 0

            if reasoning is not None:
                score += 1

            if solution is not None:
                score += 1

            scores.append(score)

        return scores

    def score_judge(self, prompts, completions, info, **kwargs):

        scores = []

        for completion, info_ in zip(completions, info):

            response = completion[0]["content"]

            reasoning, solution = self.re.extract_reasoning_solution(response)

            s = info_.get("score", 0)
            e = info_.get("detail", {})

            min_score = e.get('最低分', 0)
            zy = e.get("专业", '')

            this_score = 0

            if s < min_score:
                if reasoning and f'{s}<{min_score}' in reasoning:
                    this_score += 3.0

                if solution and f'不可以报考{zy}专业' == solution.strip():
                    this_score += 5.0

            if s >= min_score:
                if reasoning and f'{s}>{min_score}' in reasoning:
                    this_score += 3.0

                if solution and f'可以报考{zy}专业' == solution.strip():
                    this_score += 5.0

            scores.append(this_score)

        return scores

    def rank_judge(self, prompts, completions, info, **kwargs):

        scores = []

        for completion, info_ in zip(completions, info):

            response = completion[0]["content"]
            reasoning, solution = self.re.extract_reasoning_solution(response)

            r = info_.get("rank", 0)
            e = info_.get("detail", {})

            min_rank = e.get('最低位次号', 0)
            zy = e.get("专业", '')

            this_score = 0

            if r > min_rank:
                if reasoning and f'{r}>{min_rank}' in reasoning:
                    this_score += 3.0

                if solution and f'不可以报考{zy}专业' == solution.strip():
                    this_score += 5.0

            if r <= min_rank:
                if reasoning and f'{r}<{min_rank}' in reasoning:
                    this_score += 3.0

                if solution and f'可以报考{zy}专业' == solution.strip():
                    this_score += 5.0

            scores.append(this_score)

        return scores

    def task_reward(self, prompts, completions, task, info, **kwargs):

        task_ = task[0]

        if task_ == "score_judge":
            return self.score_judge(prompts, completions, info, **kwargs)

        elif task_ == "rank_judge":
            return self.rank_judge(prompts, completions, info, **kwargs)

        elif task_ == "recommend":
            return self.recommend(prompts, completions, info, **kwargs)

        elif task_ == "query_info":
            return self.query_info(prompts, completions, info, **kwargs)

        else:
            raise ValueError(f"Unknown task: {task_}")

    def build_reward(self):
        
        return [
            # self.start_with_reasoning_reward,
            # self.F1_reward,
            self.F1_bge,
            # self.check_answer
            # self.format_score,
            # self.task_reward
        ]


class ThisReward(MyReward):
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)


