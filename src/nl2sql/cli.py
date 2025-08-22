
import click
from query_engine import ExcelQueryEngine
from llm_parser import LLMQueryParser
from src.constant.inputs import inputs

import argparse

parser = argparse.ArgumentParser(description="示例：添加命令行参数")
parser.add_argument("--excel", type=str, required=False, help="test")
args = parser.parse_args()


class Cli:
    
    def __init__(self):

        self.excel = 'data/2022_23_24年浙江树人学院各省份录取情况.xlsx'
        self.engine = ExcelQueryEngine(self.excel)
        self.parser = LLMQueryParser()

    def test(self, query):
        
        pandas_query = self.parser.parse(query)
        click.echo(f"{query} -> {pandas_query}")

        try:
            result = self.engine.run_query(pandas_query)
            if result.empty:
                click.echo("❌ 没有匹配结果")
            else:
                click.echo("✅ 查询结果：")
                click.echo(result.to_string(index=False))
        except Exception as e:
            click.echo(f"执行出错: {e}")
            

def main():

    cli = Cli()

    for i in inputs:
        cli.test(i)

    # query = '2025江苏物理生物地理478分希望大吗'
    # query = '2024江苏物理最低分'

if __name__ == '__main__':
    
    main()
