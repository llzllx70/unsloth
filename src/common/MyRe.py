
import re
from src.prompt.MyPrompt import *


class MyRe:
    
    def __init__(self):

        self.format_re = re.compile(
            r"<REASONING>(.*)</REASONING>\s*<SOLUTION>(.*)</SOLUTION>",
            flags=re.DOTALL  # 关键：允许 . 匹配换行
        )

        self.reasoning_re = re.compile(
            r"<REASONING>(.*)</REASONING>",
            flags=re.DOTALL  # 关键：允许 . 匹配换行
        )

        self.solution_re = re.compile(
            rf"{solution_start}(.*){solution_end}",
            flags=re.DOTALL  # 关键：允许 . 匹配换行
        )

    def extract_reasoning_solution(self, response):

        if not response.startswith("<REASONING>"):
            response = "<REASONING>" + response

        if self.format_re.search(response) is not None:

            match = self.format_re.search(response)
            reasoning = match.group(1).strip()
            solution = match.group(2).strip()

            return reasoning, solution

        return None, None

    def extract_solution(self, response):

        if self.solution_re.search(response) is not None:

            match = self.solution_re.search(response)
            solution = match.group(1).strip()

            return solution

        return None
