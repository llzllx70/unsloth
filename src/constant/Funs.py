
from src.prompt.MyPrompt import chat_template_, system_prompt, reasoning_start

def set_tokenizer_chat_template(tokenizer):

    # Replace with out specific template:
    chat_template = chat_template_\
        .replace("'{system_prompt}'", f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")

    tokenizer.chat_template = chat_template
