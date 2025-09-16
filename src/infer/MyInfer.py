
from src.constant.Config import *
from src.common.Funs import set_tokenizer_chat_template
from src.dataset.PretrainDataset import PretrainDataset
from src.dataset.SFTDataset import SFTDataset
from src.dataset.GRPODataset import GRPODataset
from src.prompt.MyPrompt import *

from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd
import argparse
import os
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

parser = argparse.ArgumentParser(description="示例：添加命令行参数")
parser.add_argument("--task", type=str, required=False, help="test")
parser.add_argument("--model", type=str, required=False, help="Qwen3-4B-Base")
parser.add_argument("--step", type=int, required=False, help="flag")
args = parser.parse_args()


class BaseInfer:
    
    def __init__(self, model):
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def print(self, idx, input, output):

        print(f"----------------------------{idx+1} Input------------------------")
        print(f"{input}")
        print(f"----------------------------Output------------------------")
        print(f"{output}\n")    


class PretrainInfer(BaseInfer):
    
    def __init__(self):
        super().__init__(pretrain_merged_model)
        self.dataset = PretrainDataset(self.tokenizer)

    def do_infer(self):

        for idx, e in enumerate(self.dataset.test_dataset):

            text = e["prefix"]
                    
            output = self.model.generate(
                **self.tokenizer(text, return_tensors="pt").to("cuda"),
                max_new_tokens=20480,
                # do_sample=True,
                # temperature=0.01,
                use_cache=True
            )

            output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            self.print(idx, text, output=output_text)

class ForgetPretrainInfer(PretrainInfer):

    def __init__(self):
        BaseInfer.__init__(self, sft_merged_model)
        self.dataset = PretrainDataset(self.tokenizer)


class SFTInfer(BaseInfer):
    
    def __init__(self):
        super().__init__(sft_merged_model)
        self.dataset = SFTDataset(self.tokenizer)
        set_tokenizer_chat_template(self.tokenizer)

    def inner_infer(self, message):

        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,  # Must add for generation
        )

        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

        output = self.model.generate(
            **inputs,
            max_new_tokens=20480,
            do_sample=True,
            temperature=0.1,
        )

        # 输入的长度
        input_length = inputs["input_ids"].shape[1]

        # 只取生成的新 token 部分
        generated_tokens = output[0][input_length:]

        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return text, output_text

    def do_infer(self):

        ret = []

        for idx, e in enumerate(self.dataset.test_dataset):

            text, output_text = self.inner_infer(e["Messages"][:2])
            self.print(idx, text, output=output_text)

        return ret

    def default_infer(self, query):

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        text, output_text = self.inner_infer(message)

        return output_text


class GRPOInfer(BaseInfer):
    
    def __init__(self):
        super().__init__(grpo_merged_model)
        self.dataset = GRPODataset(self.tokenizer)
        set_tokenizer_chat_template(self.tokenizer)

    def do_infer(self):
        
        ret = []

        for item in self.dataset.test_dataset:
            
            text = self.tokenizer.apply_chat_template(
                item['prompt'],
                tokenize = False, 
                add_generation_prompt = False
            )

            output = self.model.generate(
                **self.tokenizer(text, return_tensors="pt").to("cuda"),
                max_new_tokens=8000,
                do_sample=True,
                temperature=0.1,
            )

            output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            print(f"Input: {text}")
            print(f"Output: {output_text}\n")    

            if self.task == 'sft':
                ret.append({
                    f'{self.task}_info': item['info'],
                    f'{self.task}_text': text,
                    f'{self.task}_output': output_text 
                })

            else:
                ret.append({
                    f'{self.task}_output': output_text 
                })

        return ret


class CompareInfer:

    def scp(self, file_):
        
        import subprocess

        # scp 命令
        cmd = [
            "scp",
            file_,
            f"double@172.16.2.4://Users/double/Downloads/"
        ]

        # 执行命令
        subprocess.run(cmd, check=True)

        print(f"文件 {file_} 已成功复制到远程服务器。")

    def do_infer(self):

        excel = 'compare_infer.xlsx'

        ret1 = BaseInfer(task='sft', model=args.model).do_grpo_infer()
        ret2 = BaseInfer(task='grpo', model=args.model).do_grpo_infer()

        ret = []

        for a, b in zip(ret1, ret2):
            ret.append({**a, **b})     

        df = pd.DataFrame(ret)

        if os.path.exists(excel):
            os.remove(excel)

        df.to_excel('compare_infer.xlsx', index=False)
        
        # 3. 用 openpyxl 打开并格式化
        wb = load_workbook(excel)
        ws = wb.active

        # 设置列宽（按需要调整）
        col_widths = {
            "A": 40,  # question
            "B": 40,  # text
            "C": 60,  # lora=False
            "D": 60,  # lora=True
        }

        for col, width in col_widths.items():
            ws.column_dimensions[col].width = width

        # 冻结首行
        ws.freeze_panes = "A2"

        # 设置首行颜色和字体
        header_fill = PatternFill(start_color="FFD966", end_color="FFD966", fill_type="solid")  # 淡黄色
        header_font = Font(bold=True, color="000000")  # 黑色加粗

        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(wrap_text=True, vertical="top")

        # 设置所有单元格自动换行 + 顶端对齐
        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")

        # 自动计算行高（根据列宽估算行数）
        for row in ws.iter_rows():
            max_lines = 1
            for cell in row:
                if cell.value:
                    col_letter = cell.column_letter
                    col_width = col_widths.get(col_letter, 10)  # 取固定列宽
                    text_length = len(str(cell.value))
                    # 按列宽估算行数
                    lines = max(1, int(text_length / (col_width * 0.9)) + 1)
                    max_lines = max(max_lines, lines)
            ws.row_dimensions[row[0].row].height = max_lines * 15  # 每行大约 15 高度

        # 保存
        wb.save(excel)

        print(f"写入完成并格式化 → {excel}")

        self.scp(excel)

if __name__ == '__main__':

    if args.task == 'pretrain':
        ret1 = PretrainInfer().do_infer()

    elif args.task == 'forget_pretrain':
        ret1 = ForgetPretrainInfer().do_infer()

    elif args.task == 'sft':
        ret1 = SFTInfer().do_infer()

    else:
        CompareInfer().do_infer()
