
from MyPrompt import chat_template_, system_prompt, reasoning_start

class BaseTrainer:

    def set_tokenizer_chat_template(self):

        # Replace with out specific template:
        chat_template = chat_template_\
            .replace("'{system_prompt}'", f"'{system_prompt}'")\
            .replace("'{reasoning_start}'", f"'{reasoning_start}'")

        self.tokenizer.chat_template = chat_template
    
    def format_print(self, query, text, output, use_lora):
        
        # print(f'\n=================lora:{use_lora}==========={query}==========')
        print(f'\n----------lora: {use_lora}-------text--------------------------')
        print(f'{text}')
        print(f'-----------------output------------------------')
        print(f'{output}')
        # print(f'==================End of output====================\n')
        
