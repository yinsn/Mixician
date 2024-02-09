from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import pandas as pd
from pydantic import BaseModel


class BaseBalancerConfig(BaseModel):
    request_id_column: str = "request_id"
    overall_score_column: str = "overall_score"
    category_column: str = "category"
    max_rows: Optional[int] = None


class BaseBalancer(metaclass=ABCMeta):
    def __init__(self, dataframe: pd.DataFrame, config: Optional[Dict] = None) -> None:
        self.config = BaseBalancerConfig(**(config or {}))
        self.request_id_column = self.config.request_id_column
        self.overall_score_column = self.config.overall_score_column
        self.category_column = self.config.category_column
        self.max_rows = self.config.max_rows
        self.dataframe = (
            dataframe.head(self.max_rows) if self.max_rows is not None else dataframe
        )
