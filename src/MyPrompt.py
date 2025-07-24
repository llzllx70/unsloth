
reasoning_start = "<REASONING>" # Acts as <think>
reasoning_end   = "</REASONING>"   # Acts as </think>
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
reasoning about the problem and provide your working out.
Place your reason between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

system_prompt_ok = \
f"""You are given a problem.
Provide your answer between {solution_start}{solution_end}.
For example: {solution_start}x{solution_end}
"""

chat_template_ = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

r_strawberry = """<REASONING> To find how many **r**'s are in the word **strawberry**, we can go through each letter one by one and count the occurrences of the letter **r**.
The word is: s t r a w b e r r y

Now, let's count:
s → not r
t → not r
r → 1st r
a → not r
w → not r
b → not r
e → not r
r → 2nd r
r → 3rd r
y → not r

There are 3 r's in total.
</REASONING>"""

r_good = """<REASONING> We are asked to count how many times the letter **o** appears in the word **good**.
The word is: g o o d

Now, let's go through each letter:
g → not o
o → 1st o
o → 2nd o
d → not o

So there are 2 occurrences of the letter o in the word good.
</REASONING>"""

r_deepseek = """<REASONING> We are asked to count how many times the letter **e** appears in the word **deepseek**.
The word is: d e e p s e e k

Now, examine each letter:
d → not e
e → 1st e
e → 2nd e
p → not e
s → not e
e → 3rd e
e → 4th e
k → not e

There are 4 instances of the letter e in the word deepseek.
</REASONING>"""

r_FastLanguageModel = """
<REASONING> To determine how many times the letter "a" appears in the string "FastLanguageModel", we will go through each character and count the lowercase letter 'a'.
String: FastLanguageModel
Characters:
F → not 'a'
a → 1
s → not 'a'
t → not 'a'
L → not 'a'
a → 2
n → not 'a'
g → not 'a'
u → not 'a'
a → 3
g → not 'a'
e → not 'a'
M → not 'a'
o → not 'a'
d → not 'a'
e → not 'a'
l → not 'a'

There are 3 occurrences of the letter 'a' in total.
</REASONING>
<SOLUTION>3</SOLUTION>"""

MyTrainDataset = [
    ("How many r's are in strawberry?", '3', r_strawberry),
    ("How many o's are in good?", '2', r_good),
    ("How many e's are in deepseek?", '4', r_deepseek),
    ("How many a's are in FastLanguageModel?", '3', r_FastLanguageModel)
]

MyTrainDataset_ok = [
    ("Random chose one num from [3, 4, 5, 6], which num do you get this time?", '5', r_strawberry),
    ("Random chose one num from [3, 4, 5, 6], which num do you get this time?", '5', r_strawberry),
    ("Random chose one num from [3, 4, 5, 6], which num do you get this time?", '5', r_strawberry),
    ("Random chose one num from [3, 4, 5, 6], which num do you get this time?", '5', r_strawberry),
    ("Random chose one num from [3, 4, 5, 6], which num do you get this time?", '5', r_strawberry),
]

MyTestDataset = [
    ("How many a's are in string FastLanguageModel?", '4')
]
