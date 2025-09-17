
import jieba
from src.common.MyRe import MyRe

class MyReward:

    def __init__(self, tokenizer=None):

        if tokenizer is None:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = tokenizer.eos_token

        self.re = MyRe()

    def score_print(self, scores, flag):

        print(f'\n=================score:{flag}=====================')
        print(f'{scores}')

    def start_with_reasoning_reward(self, completions, **kwargs):

        scores = []

        for completion in completions:
            response = completion[0]["content"]
            score = 10 if response.strip().startswith("<REASONING>") else 0.0
            scores.append(score)

        self.score_print(scores=scores,flag='0. start_with_reasoning_reward')
        return scores

    def F1_reward(self, completions, query, out, **kwargs):

        scores = []

        for idx, completion in enumerate(completions):

            pred = completion[0]['content']
            truth = out[idx]

            pred_tokens = list(jieba.lcut(pred))
            truth_tokens = list(jieba.lcut(truth))

            common = set(pred_tokens) & set(truth_tokens)
            common_count = sum(min(pred_tokens.count(t), truth_tokens.count(t)) for t in common)

            if common_count == 0:
                scores.append(0.0)

            else:
                precision = common_count / len(pred_tokens)
                recall = common_count / len(truth_tokens)
                scores.append(2 * precision * recall / (precision + recall))
            
        self.score_print(scores=scores, flag='2. F1_reward')
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
            self.F1_reward,
            # self.check_answer
            # self.format_score,
            # self.task_reward
        ]


class ThisReward(MyReward):
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)


