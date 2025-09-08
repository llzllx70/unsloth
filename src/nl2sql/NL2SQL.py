
import click
import traceback
from src.nl2sql.PandasQuery import PandasEngine
from src.nl2sql.LLMApi import LLMApi
from src.constant.inputs import inputs
from src.common.MyRe import MyRe

import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="示例：添加命令行参数")
parser.add_argument("--excel", type=str, required=False, help="test")
args = parser.parse_args()


class NL2SQL:
    
    def __init__(self):

        self.excel = 'data/2022_23_24年浙江树人学院各省份录取情况.xlsx'
        self.sft_data = 'data/sft_data.xlsx'
        self.pandas_engine = PandasEngine(self.excel)
        self.qwen_api = LLMApi()
        self.re = MyRe()

    def test(self, query):
        
        pandas_query = self.qwen_api.nl2sql(query)
        click.echo(f"{query} -> {pandas_query}")

        try:
            result = self.pandas_engine.run_query(pandas_query)
            if result.empty:
                click.echo("❌ 没有匹配结果")
            else:
                click.echo("✅ 查询结果：")
                click.echo(result.to_string(index=False))
        except Exception as e:
            click.echo(f"执行出错: {e}")

    def build_sft(self):

        def f(q):

            sql = self.qwen_api.nl2sql(q)

            try:
                result = self.pandas_engine.run_query(sql)

                if not result.empty:

                    md = result.to_markdown(index=False)
                    r = self.qwen_api.answer(q, md)
                    reasoning, solution = self.re.extract_reasoning_solution(r)

                    click.echo(f"{q} -- {sql} ok")
                    return (q, sql, md, reasoning, solution)

                else:
                    click.echo(f"{q} -- {sql} none")
                    return (q, sql, None, None, None)

            except Exception as e:
                traceback.print_exc()  # 打印完整的异常堆栈

                click.echo(f"{q} -- {sql} error")
                return (q, sql, None, None, None)

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(f, inputs))

        df = pd.DataFrame(results, columns=['query', 'sql', 'result', 'reasoning', 'solution'])
        df.to_excel(self.sft_data, index=False)


def main():

    nl2sql = NL2SQL()
    nl2sql.build_sft()

    # query = '2025江苏物理生物地理478分希望大吗'
    # query = '2024江苏物理最低分'

if __name__ == '__main__':
    
    main()
