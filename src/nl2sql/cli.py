
import click
from query_engine import ExcelQueryEngine
from llm_parser import LLMQueryParser

import argparse

parser = argparse.ArgumentParser(description="示例：添加命令行参数")
parser.add_argument("--excel", type=str, required=False, help="test")
args = parser.parse_args()

def main():

    """Excel 检索 CLI 工具"""
    engine = ExcelQueryEngine(args.excel)
    parser = LLMQueryParser()

    query = '江苏物理生物地理478分希望大吗'
    
    pandas_query = parser.parse(query)
    click.echo(f"👉 解析得到的 pandas 查询语句: {pandas_query}")

    try:
        result = engine.run_query(pandas_query)
        if result.empty:
            click.echo("❌ 没有匹配结果")
        else:
            click.echo("✅ 查询结果：")
            click.echo(result.to_string(index=False))
    except Exception as e:
        click.echo(f"执行出错: {e}")


if __name__ == '__main__':
    
    main()