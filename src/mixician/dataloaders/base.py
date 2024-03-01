import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import pandas as pd
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class BaseDataLoaderConfig(BaseModel):
    """Configuration model for data loaders.

    This class defines the configuration settings for data loaders, including file
    path, file name, file type, and maximum rows to be loaded. It utilizes Pydantic's
    BaseModel for data validation and settings management.

    Attributes:
        file_path (Optional[str]): The path to the file to be loaded. Defaults to None.
        file_name (Optional[str]): The name of the file to be loaded. Defaults to None.
        file_type (str): The type of the file to be loaded. Defaults to 'pkl' (pickle).
        max_rows (Optional[int]): The maximum number of rows to load from the file.
                                  Defaults to None, indicating no limit.
    """

    file_path: Optional[str] = None
    file_name: Optional[str] = None
    file_type: str = "pkl"
    max_rows: Optional[int] = None


class BaseDataLoader(metaclass=ABCMeta):
    """
    Extract from 'Lova' project (https://github.com/yinsn/Lova/blob/develop/src/lova/dataloaders/base.py).

    This class provides a foundation for data loaders, initializing with a
    configuration dictionary and setting up essential attributes for data
    handling.

    Attributes:
        config (BaseDataLoaderConfig): Configuration object holding settings for data loading.
        file_path (str): Path to the file to be loaded.
        file_name (str): Name of the file to be loaded.
        file_type (str): Type of the file to be loaded (e.g., CSV, Pickle).
        max_rows (int): Maximum number of rows to load from the file.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        """Initializes the data loader with the specified configuration.

        Args:
            config (Optional[Dict]): Configuration dictionary to initialize the BaseDataLoaderConfig object.
                                     If None, defaults to an empty dictionary.
        """
        logger.info("Initializing data loader...")
        self.config = BaseDataLoaderConfig(**(config or {}))
        self.file_path = self.config.file_path
        self.file_name = self.config.file_name
        self.file_type = self.config.file_type
        self.max_rows = self.config.max_rows

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
