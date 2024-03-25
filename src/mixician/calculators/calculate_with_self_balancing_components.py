import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from IPython.display import Latex, display

from .calculate_with_components import (
    LogarithmPCACalculator,
    LogarithmPCACalculatorConfig,
)
from .limit_upper_bound import find_top_percentile_value

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class SelfBalancingLogarithmPCACalculatorConfig(LogarithmPCACalculatorConfig):
    """Configuration for the SelfBalancingLogarithmPCACalculator.

    Attributes:
        upper_bound_3sigma (float): Upper bound defined as 3 sigma from the mean. Defaults to 10.
    """

    upper_bound_3sigma: float = 10


class SelfBalancingLogarithmPCACalculator(LogarithmPCACalculator):
    """Calculator for self-balancing logarithm PCA weights.

    Inherits from LogarithmPCACalculator to adjust weight calculations for
    a target distribution.

    Attributes:
        config (SelfBalancingLogarithmPCACalculatorConfig): Configuration for the calculator.
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
        self.upper_bound_3sigma = self.config.upper_bound_3sigma
        self.upper_bound_3sigma_in_log10 = np.log10(self.upper_bound_3sigma)
        self.target_distribution_mean = self.upper_bound_3sigma_in_log10 / 2.0
        self.calculte_balanced_weights()

    def _calculate_first_order_weights(self) -> None:
        """Calculates first-order weights for the PCA transformation."""
        self.first_order_weights = 10.0 ** (
            self.target_distribution_mean / sum(self.power_weights)
            - self.logarithm_data_means
        )

    def _calculate_power_weights(self) -> None:
        """Calculates power weights based on projected data standard deviation."""
        projected_data = np.dot(self.data_normalized, self.sorted_eigenvectors)
        self.projected_data_3sigma_std = np.std(projected_data, axis=0) * 6.0
        self.power_weights = (
            np.squeeze(self.sorted_eigenvectors) / self.projected_data_3sigma_std
        )
        if sum(self.power_weights) < 0:
            self.power_weights = -self.power_weights

    def calculte_balanced_weights(self) -> None:
        """Calculates and applies balanced weights to the PCA components."""
        self._calculate_power_weights()
        self._calculate_first_order_weights()
        self._calculate_cumulative_product_scores()
        self.tune_upper_bound(self.upper_bound_3sigma)

    def _calculate_cumulative_product_scores(self) -> None:
        """Calculates the cumulative product of the scores."""
        self.cumulative_product_scores = 1 + np.multiply(
            self.data, self.first_order_weights
        )
        self.cumulative_product_scores = (
            self.cumulative_product_scores**self.power_weights
        )
        self.cumulative_product_scores = np.prod(self.cumulative_product_scores, axis=1)

    def _calculate_percentile_weights(self, scaling_factor: float = 1.0) -> float:
        """
        Adjusts power weights by a scaling factor and calculates the upper bound based on the adjusted weights.

        This method temporarily adjusts the power weights of the object by a given scaling factor, recalculates
        dependent metrics, and then calculates the upper bound of the cumulative product scores at a very
        small percentile. After calculation, the power weights are reset to their original values.

        Args:
            scaling_factor (float): A scaling factor to adjust the power weights, defaulted to 1.0.

        Returns:
            float: The calculated upper bound value at the specified small percentile of cumulative product scores.

        """
        self.initial_power_weights = self.power_weights
        self.power_weights = self.power_weights * scaling_factor
        self._calculate_first_order_weights()
        self._calculate_cumulative_product_scores()
        upper_bound = find_top_percentile_value(self.cumulative_product_scores, 0.00135)
        self.power_weights = self.initial_power_weights
        return upper_bound

    def tune_upper_bound(
        self,
        target_upper_bound: float,
        low: float = 0.001,
        high: float = 5.0,
        tolerance: float = 0.01,
    ) -> None:
        """
        Tunes the scaling factor to achieve a desired upper bound for the cumulative product scores.

        This method performs a binary search within a specified range to find a scaling factor that,
        when applied, results in a calculated upper bound close to a target value. It aims to adjust
        the scaling factor such that the absolute difference between the calculated upper bound and
        the target upper bound is within a given tolerance.

        Args:
            target_upper_bound (float): The desired upper bound value to tune for.
            low (float): The lower bound of the scaling factor range for the binary search, defaults to 0.001.
            high (float): The upper bound of the scaling factor range for the binary search, defaults to 5.0.
            tolerance (float): The acceptable tolerance level for the difference between the calculated
                upper bound and the target upper bound, defaults to 0.01.

        """
        logger.info(f"Tuning upper bound to {target_upper_bound}...")
        while low <= high:
            mid = (low + high) / 2.0
            upper_bound = self._calculate_percentile_weights(mid)
            if abs(upper_bound - target_upper_bound) <= tolerance:
                self.power_weights = self.initial_power_weights * low
                self._calculate_first_order_weights()
                self._calculate_cumulative_product_scores()
                break
            elif upper_bound < target_upper_bound:
                low = mid
            else:
                high = mid

    def plot_self_balancing_projected_distribution(self) -> None:
        """Plots the projected data distribution after applying logarithm PCA and balancing weights."""
        if self.cumulative_product_scores is None:
            self._calculate_cumulative_product_scores()
        self.viewer_instance.plot_logarithm_array_distribution(
            scores=np.asarray(self.cumulative_product_scores),
            legend=r"$\log_{10}{(\text{cumulative\_product\_scores})}$",
        )

    def update(self, pca_weights: np.ndarray) -> None:
        """Updates the PCA weights with the provided weights."""
        self.update_pca_weights(pca_weights)
        self.calculte_balanced_weights()

    def get_weights(self) -> None:
        """
        Logs the weights and importance of each score column associated with this instance.

        This method combines various weight metrics (PCA default weights, first order weights, and power weights)
        for each scoring column, formats them into a human-readable string, and logs the information using the
        class's logger.

        The method temporarily modifies NumPy's print options to suppress scientific notation for better readability.
        """
        default_options = np.get_printoptions()
        np.set_printoptions(suppress=True)
        messages = []
        for column, importance, fo_weight, p_weight in zip(
            self.selected_columns,
            np.asarray(self.pca_default_weights),
            self.first_order_weights,
            self.power_weights,
        ):
            message = (
                f"\n\ncolumn: \t\t\t {column}\n"
                f"variance importance: \t\t {importance}\n"
                f"first_order_weights: \t\t {fo_weight}\n"
                f"power_weights: \t\t\t {p_weight}\n"
            )
            messages.append(message)

        full_message = "\n".join(messages)
        self.results = full_message
        np.set_printoptions(**default_options)

    def show_equation(self) -> None:
        """
        Displays the equation in LaTeX format, using instance variables for weights and score columns.
        """
        a = self.first_order_weights
        b = self.power_weights
        x = self.selected_columns

        a_formatted = [f"{ai:.6g}" for ai in a]
        b_formatted = [f"{bi:.6g}" for bi in b]
        x_formatted = [f"\\text{{{item}}}".replace("_", "\_") for item in x]

        formula_parts = [
            f"(1 + {a_i} \cdot {x_i})^{{{b_i}}}"
            for a_i, b_i, x_i in zip(a_formatted, b_formatted, x_formatted)
        ]
        formula = " \\times ".join(formula_parts)

        latex_formula = f"\prod_{{i=1}}^{{{len(a)}}} " + formula
        self.latex_formula = latex_formula
        display(Latex(f"$ {self.latex_formula} $"))

    def show_results(self) -> None:
        self.get_weights()
        self.show_equation()
        logger.info(self.results)
