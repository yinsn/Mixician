from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel


class BaseCalculatorConfig(BaseModel):

    score_columns: List[str]
    total_score_column: str = "total_score"
    max_rows: Optional[int] = None


class BaseCalculator:

    def __init__(self, dataframe: pd.DataFrame, config: Optional[Dict] = None) -> None:
        self.config = BaseCalculatorConfig(**(config or {}))
        self.score_columns = self.config.score_columns
        self.total_score_column = self.config.total_score_column
        self.max_rows = self.config.max_rows
        self.dataframe = (
            dataframe.head(self.max_rows) if self.max_rows is not None else dataframe
        )

    def calculate(self) -> None:
        pass
