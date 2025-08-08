
from constant.Funs import set_tokenizer_chat_template

class BaseTrainer:

    def format_print(self, query, text, output, use_lora):
        
        # print(f'\n=================lora:{use_lora}==========={query}==========')
        print(f'\n----------lora: {use_lora}-------text--------------------------')
        print(f'{text}')
        print(f'-----------------output------------------------')
        print(f'{output}')
        # print(f'==================End of output====================\n')
        
