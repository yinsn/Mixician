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
    """Configuration for the SelfBalancingLogarithmPCACalculator.

    Attributes:
        target_distribution_mean (float): Target mean for the distribution after balancing. Defaults to -.5.
        upper_bound_3sigma (float): Upper bound defined as 3 sigma from the mean. Defaults to 1.0.
    """

    target_distribution_mean: float = 0.5
    upper_bound_3sigma: float = 1.0


class SelfBalancingLogarithmPCACalculator(LogarithmPCACalculator):
    """Calculator for self-balancing logarithm PCA weights.

    Inherits from LogarithmPCACalculator to adjust weight calculations for
    a target distribution.

    Attributes:
        config (SelfBalancingLogarithmPCACalculatorConfig): Configuration for the calculator.
        target_distribution_mean (float): Target mean for the balanced distribution.
        upper_bound_3sigma (float): Defined upper bound as 3 standard deviations from the mean.
    """

    def __init__(self, dataframe: pd.DataFrame, config: Optional[Dict]) -> None:
        """Initializes the self-balancing logarithm PCA calculator.

        Args:
            dataframe (pd.DataFrame): The input data for PCA calculation.
            config (Optional[Dict[str, Any]]): Configuration dictionary, which is unpacked into SelfBalancingLogarithmPCACalculatorConfig.
        """
        super().__init__(dataframe, config)
        self.config = SelfBalancingLogarithmPCACalculatorConfig(**(config or {}))
        self.target_distribution_mean = self.config.target_distribution_mean
        self.upper_bound_3sigma = self.config.upper_bound_3sigma

    def _calculate_first_order_weights(self) -> None:
        """Calculates first-order weights for the PCA transformation."""
        self.first_order_weights = 10.0 ** (
            self.target_distribution_mean / sum(self.power_weights)
            - self.logarithm_data_means / np.asarray(self.config.pca_default_weights)
        )

    def _calculate_power_weights(self) -> None:
        """Calculates power weights based on projected data standard deviation."""
        projected_data = np.dot(self.data_normalized, self.sorted_eigenvectors)
        self.projected_data_3sigma_std = np.std(projected_data, axis=0) * 6.0
        self.power_weights = (
            np.squeeze(self.sorted_eigenvectors) / self.projected_data_3sigma_std
        )

    def calculte_balanced_weights(self) -> None:
        """Calculates and applies balanced weights to the PCA components."""
        self._calculate_power_weights()
        self._calculate_first_order_weights()

    def _calculate_cumulative_product_scores(self) -> None:
        """Calculates the cumulative product of the scores."""
        logger.info("Calculating cumulative product of scores...")
        self.cumulative_product_scores = 1 + np.multiply(
            self.data, self.first_order_weights
        )
        self.cumulative_product_scores = (
            self.cumulative_product_scores**self.power_weights
        )
        self.cumulative_product_scores = np.prod(self.cumulative_product_scores, axis=1)

    def plot_self_balancing_projected_distribution(self) -> None:
        """Plots the projected data distribution after applying logarithm PCA and balancing weights."""
        self._calculate_cumulative_product_scores()
        self.viewer_instance.plot_array_distribution(
            scores=np.log10(self.cumulative_product_scores),
            legend="Projected Data",
        )
