import logging
import os
from typing import Dict, Optional

import pandas as pd

from .base import BaseDataLoader, BaseDataLoaderConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class DataFrameLoader(BaseDataLoader):
    """Data loader for loading data into pandas DataFrame.

    This class inherits from BaseDataLoader and implements methods for loading
    data from different file formats into pandas DataFrames, currently supporting CSV files.

    Attributes:
        config (Optional[Dict]): Configuration options for loading.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
    ) -> None:
        super().__init__(config=config)
        self.config = BaseDataLoaderConfig(**(config or {}))

    def _load_from_csv(self) -> pd.DataFrame:
        """Loads data from a CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        logger.info(f"Loading csv data from {self.file_path}...")
        file_url = os.path.join(str(self.file_path), str(self.file_name))
        return pd.read_csv(file_url + ".csv", nrows=self.max_rows)

    @staticmethod
    def _column_name_spliting(
        dataframe: pd.DataFrame, delimiter: str = "."
    ) -> pd.DataFrame:
        """Splits column names in the DataFrame using a specified delimiter.

        Args:
            dataframe (pd.DataFrame): The DataFrame whose columns are to be split.
            delimiter (str): The delimiter to use for splitting column names.

        Returns:
            pd.DataFrame: The DataFrame with split column names.
        """
        logger.info(f"Splitting column names by delimiter '{delimiter}'...")
        columns = []
        for column in dataframe.columns:
            columns.append(column.split(delimiter)[-1])
        dataframe.columns = pd.Index(columns)
        return dataframe

    def load_data(self) -> pd.DataFrame:
        """Loads data based on the file type specified during initialization.

        After loading, it processes the column names by splitting them based on a predefined delimiter.

        Returns:
            pd.DataFrame: The loaded and processed data as a pandas DataFrame.
        """
        if self.file_type == "csv":
            dataframe = self._load_from_csv()
        dataframe = self._column_name_spliting(dataframe)
        return dataframe
