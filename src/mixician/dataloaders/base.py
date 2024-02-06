from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import pandas as pd


class BaseDataLoader(metaclass=ABCMeta):
    """
    Extract from 'Lova' project (https://github.com/yinsn/Lova/blob/develop/src/lova/dataloaders/base.py).
    Initialize the BaseDataLoader object.

    This abstract base class defines the structure for data loaders. Implementations of this class should
    override the `load_data` method to load data from a specified source.

    Args:
        file_path (str, optional): The path to the directory where the file is located.
        file_name (str, optional): The name of the file without extension.
        file_type (str, optional): The type of the file. Defaults to 'pkl'.
        max_rows (Optional[int], optional): The maximum number of rows to load from the file. Defaults to None.
        config (Optional[Dict], optional): A dictionary containing the configuration for the data loader. Defaults to None.
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        file_type: str = "pkl",
        max_rows: Optional[int] = None,
        config: Optional[Dict] = None,
    ) -> None:
        if config is not None:
            self.file_path = config.get("file_path", None)
            self.file_name = config.get("file_name", None)
            self.file_type = config.get("file_type", "pkl")
            self.max_rows = config.get("max_rows", None)
        else:
            self.file_path = file_path
            self.file_name = file_name
            self.file_type = file_type
            self.max_rows = max_rows

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Abstract method to load data into a pandas DataFrame.

        Subclasses must implement this method. It should read data from a source, process it as necessary,
        and return it as a pandas DataFrame.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
