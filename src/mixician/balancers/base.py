from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import pandas as pd
from pydantic import BaseModel


class BaseBalancerConfig(BaseModel):
    """Configuration for base balancer.

    Attributes:
        request_id_column (str): Column name for request ID.
        overall_score_column (str): Column name for overall score.
        category_column (str): Column name for category.
        max_rows (Optional[int]): Maximum number of rows to consider from the dataframe.
    """

    request_id_column: str = "request_id"
    overall_score_column: str = "overall_score"
    category_column: str = "category"
    max_rows: Optional[int] = None


class BaseBalancer(metaclass=ABCMeta):
    """Abstract base class for balancers.

    This class provides a template for implementing different types of balancers.

    Attributes:
        dataframe (pd.DataFrame): The dataframe to balance.
        config (Optional[Dict]): Optional dictionary to override default configuration.
    """

    def __init__(self, dataframe: pd.DataFrame, config: Optional[Dict] = None) -> None:
        """Initializes the BaseBalancer with a dataframe and optional configuration overrides.

        Args:
            dataframe (pd.DataFrame): The dataframe to be balanced.
            config (Optional[Dict]): Configuration overrides.
        """
        self.config = BaseBalancerConfig(**(config or {}))
        self.request_id_column = self.config.request_id_column
        self.overall_score_column = self.config.overall_score_column
        self.category_column = self.config.category_column
        self.max_rows = self.config.max_rows
        self.dataframe = (
            dataframe.head(self.max_rows) if self.max_rows is not None else dataframe
        )

    @abstractmethod
    def evaluate(self) -> None:
        """Evaluates or processes the dataframe according to the balancer's logic.

        This method must be implemented by subclasses.
        """
        pass
