from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel


class BaseCalculatorConfig(BaseModel):
    """Configuration for BaseCalculator.

    Attributes:
        score_columns (List[str]): A list of column names in the dataframe that are used for scoring.
        total_score_column (str): The name of the column to store the total score. Defaults to "total_score".
        max_rows (Optional[int]): The maximum number of rows to process from the dataframe. If None, all rows are processed. Defaults to None.
    """

    score_columns: List[str]
    total_score_column: str = "total_score"
    max_rows: Optional[int] = None


class BaseCalculator(metaclass=ABCMeta):
    """A base class for creating calculators that operate on pandas DataFrames.

    This class provides the basic framework for setting up a calculator, including configuration
    and initialization with a dataframe. Subclasses must implement the `calculate` method to
    perform specific calculations.

    Attributes:
        dataframe (pd.DataFrame): The dataframe on which calculations will be performed.
        config (BaseCalculatorConfig): Configuration for the calculator, encapsulating score columns,
            total score column name, and optional row limit.
        score_columns (List[str]): List of column names to be used for scoring, derived from config.
        total_score_column (str): Column name where the total score will be stored, derived from config.
        max_rows (Optional[int]): Maximum number of rows to be processed, derived from config.

    Args:
        dataframe (pd.DataFrame): The pandas DataFrame to be used for calculations.
        config (Optional[Dict]): A dictionary of configuration options. Defaults to None, in which case
            default configuration is used.
    """

    def __init__(self, dataframe: pd.DataFrame, config: Optional[Dict] = None) -> None:
        self.config = BaseCalculatorConfig(**(config or {}))
        self.score_columns = self.config.score_columns
        self.total_score_column = self.config.total_score_column
        self.max_rows = self.config.max_rows
        self.dataframe = (
            dataframe.head(self.max_rows) if self.max_rows is not None else dataframe
        )

    @abstractmethod
    def calculate(self) -> None:
        """Performs calculations on the dataframe."""
        pass
