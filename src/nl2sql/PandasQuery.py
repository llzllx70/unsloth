
import pandas as pd

class PandasEngine:

    def __init__(self, excel_path: str):

        self.df = pd.read_excel(excel_path)

    def run_query(self, query: str) -> pd.DataFrame:
        try:
            result = self.df.query(query)
            return result
        except Exception as e:
            raise RuntimeError(f"查询执行失败: {e}")
