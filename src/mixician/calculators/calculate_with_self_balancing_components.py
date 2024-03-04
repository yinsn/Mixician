import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .calculate_with_components import (
    LogarithmPCACalculator,
    LogarithmPCACalculatorConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class SelfBalancingLogarithmPCACalculatorConfig(LogarithmPCACalculatorConfig):
    target_distribution_mean: float = 5.0
    upper_bound_3sigma: float = 10.0


class SelfBalancingLogarithmPCACalculator(LogarithmPCACalculator):
    def __init__(self, dataframe: pd.DataFrame, config: Optional[Dict]) -> None:
        super().__init__(dataframe, config)
        self.config = SelfBalancingLogarithmPCACalculatorConfig(**(config or {}))
        self.target_distribution_mean = self.config.target_distribution_mean
        self.upper_bound_3sigma = self.config.upper_bound_3sigma

    def _calculate_first_order_weights(self) -> None:
        self.first_order_weights = 10.0 ** (-self.logarithm_data_means) - 1.0
        self.first_order_weights = (
            self.first_order_weights / self.target_distribution_mean
        )

    def _calculate_power_weights(self) -> None:
        projected_data = np.dot(self.data_normalized, self.sorted_eigenvectors)
        self.projected_data_3sigma_std = np.std(projected_data, axis=0) * 6.0
        self.power_weights = self.sorted_eigenvectors / self.projected_data_3sigma_std

    def calculte_balanced_weights(self) -> None:
        self._calculate_first_order_weights()
        self._calculate_power_weights()

    def plot_self_balancing_projected_distribution(self) -> None:
        """Plots the projected data after Logarithm PCA."""
        self.viewer_instance.plot_array_distribution(
            scores=self.projected_data / self.projected_data_3sigma_std
            + self.target_distribution_mean,
            legend="Projected Data",
        )
