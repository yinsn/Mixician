import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseBalancer, BaseBalancerConfig
from .compute_divergence import jensen_shannon_divergence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class ProbabilisticBalancerConfig(BaseBalancerConfig):
    """Configuration for ProbabilisticBalancer.

    Attributes:
        top_n_exposure (int): Number of top items to consider for exposure calculation.
        target_ratios (Dict[str, float]): Target ratios for each category.
    """

    top_n_exposure: int = 5
    target_ratios: Dict = {}


class ProbabilisticBalancer(BaseBalancer):
    """Probabilistic Balancer for adjusting item exposures based on target ratios.

    Attributes:
        dataframe (pd.DataFrame): The dataframe to balance.
        config (Optional[Dict]): Configuration dictionary that can be converted to ProbabilisticBalancerConfig.
    """

    def __init__(self, dataframe: pd.DataFrame, config: Optional[Dict] = None) -> None:
        """Initializes the ProbabilisticBalancer with a dataframe and optional config."""
        super().__init__(
            dataframe=dataframe,
            config=config,
        )
        self.config = ProbabilisticBalancerConfig(**(config or {}))
        self.top_n_exposure = self.config.top_n_exposure
        self.target_ratios = self.config.target_ratios
        self._initialize_target_ratios()

    def _initialize_target_ratios(self) -> None:
        """Initializes target ratios based on the categories present in the dataframe."""
        logger.info("Initializing target ratios...")
        self.compute_exposure_ratios()
        if self.target_ratios == {}:
            ratio = 1 / len(self.exposure_ratios)
            for category, _ in self.exposure_ratios.to_dict():
                self.target_ratios[category] = ratio
        else:
            target_ratios_list: List = []
            for category, _ in self.target_ratios.items():
                target_ratios_list.append(self.target_ratios[category])
            self.target_ratios_vector = np.asarray(target_ratios_list)

    def compute_exposure_ratios(self) -> None:
        """Computes the exposure ratios of categories within the top N exposures."""
        sorted_dataframe = self.dataframe.sort_values(
            [self.request_id_column, self.overall_score_column], ascending=[True, False]
        )
        truncated_dataframe = (
            sorted_dataframe.groupby(self.request_id_column)
            .head(self.top_n_exposure)
            .reset_index(drop=True)
        )
        self.exposure_ratios = (
            truncated_dataframe.groupby(self.category_column).size()
            / truncated_dataframe.groupby(self.category_column).size().sum()
        )
        self.exposure_ratios_vector = self.exposure_ratios.to_numpy()
        self.exposure_ratios_dict = self.exposure_ratios.to_dict()

    def evaluate(self) -> None:
        """Evaluates the current setup by calculating the Jensen-Shannon divergence."""
        self.js_divergence = jensen_shannon_divergence(
            self.exposure_ratios_vector, self.target_ratios_vector
        )
