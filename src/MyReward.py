
import re

from MyPrompt import *

class MyReward:

    def __init__(self, tokenizer=None):

        if tokenizer is None:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = tokenizer.eos_token

        self.format_re = self.format_re()

    def reasoning_re(self):
        
        return re.compile(
            r"<REASONING>(.*)</REASONING>",
            flags=re.DOTALL  # 关键：允许 . 匹配换行
        )

    def solution_re(self):

        return re.compile(
            rf"{solution_start}(.*){solution_end}",
            flags=re.DOTALL  # 关键：允许 . 匹配换行
        )

    def score_print(self, scores, flag):

        print(f'\n=================score:{flag}=====================')
        print(f'{scores}')

    def format_re(self):
        
        return re.compile(
            r"<REASONING>(.*)</REASONING>\s*<SOLUTION>(.*)</SOLUTION>",
            flags=re.DOTALL  # 关键：允许 . 匹配换行
        )
        
    def match_reasoning(self, completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            if self.format_re.search(response) is not None: 
                breakpoint()
                score += 3.0
            scores.append(score)

        self.score_print(scores=scores,flag='1. match_format_exactly')
        return scores
    
    def start_with_reasoning_reward(self, completions, **kwargs):

        scores = []

        for completion in completions:
            response = completion[0]["content"]
            score = 10 if response.strip().startswith("<REASONING>") else 0.0
            scores.append(score)

        self.score_print(scores=scores,flag='0. start_with_reasoning_reward')
        return scores

    def F1_reward(self, completions, ref_reason, **kwargs):

        scores = []

        for idx, completion in enumerate(completions):

            pred_reason = completion[0]['content']

            pred_tokens = set(pred_reason.strip().split())
            ref_tokens = set(ref_reason[idx].strip().split())

            overlap = pred_tokens & ref_tokens
            if not overlap:
                return 0.0
            precision = len(overlap) / len(pred_tokens)
            recall = len(overlap) / len(ref_tokens)
            scores.append(2 * precision * recall / (precision + recall))
            
        self.score_print(scores=scores,flag='2. F1_reward')
        return scores

    def check_answer(self, prompts, completions, answer, **kwargs):

        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := self.format_re.search(r)) is not None else None \
            for r in responses
        ]

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

    def build_reward(self):
        
        return [
            # self.match_format_exactly,
            self.start_with_reasoning_reward,
            # self.F1_reward,
            self.check_answer
        ]