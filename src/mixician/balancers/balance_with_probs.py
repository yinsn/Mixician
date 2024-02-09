from typing import Dict, Optional

import pandas as pd

from .base import BaseBalancer, BaseBalancerConfig


class ProbabilisticBalancerConfig(BaseBalancerConfig):
    top_n_exposure: int = 5
    target_ratios: Optional[Dict] = None


class ProbabilisticBalancer(BaseBalancer):
    def __init__(self, dataframe: pd.DataFrame, config: Optional[Dict] = None) -> None:
        super().__init__(
            dataframe=dataframe,
            config=config,
        )
        self.config = ProbabilisticBalancerConfig(**(config or {}))
        self.top_n_exposure = self.config.top_n_exposure

    def compute_exposure_ratio(self) -> pd.Series:
        sorted_dataframe = self.dataframe.sort_values(
            [self.request_id_column, self.overall_score_column], ascending=[True, False]
        )
        truncated_dataframe = (
            sorted_dataframe.groupby(self.request_id_column)
            .head(self.top_n_exposure)
            .reset_index(drop=True)
        )
        exposure_ratio = (
            truncated_dataframe.groupby(self.category_column).size()
            / truncated_dataframe.groupby(self.category_column).size().sum()
        )
        return exposure_ratio
