

from reward.MyReward import *


class MyTest:
    
    def f1(self):
        
        reward = MyReward()
        pattern = reward.build_match_format()

        # 测试输入
        test_inputs = [
            (
                "xx<REASONING>Step-by-step thinking.</REASONING>\n<SOLUTION>Final answer.</SOLUTION>",
                True
            )
        ]

        for i, (text, expected) in enumerate(test_inputs):
            match = pattern.match(text)
            result = bool(match)
            print(f"Test case {i+1}: {'Passed' if result == expected else 'Failed'}")
            if result:
                print("  Reasoning:", match.group(1).strip())
                print("  Solution :", match.group(2).strip())



MyTest().f1()
