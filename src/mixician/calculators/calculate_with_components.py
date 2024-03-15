import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..visualization import DistributionViewer
from .base import BaseCalculator, BaseCalculatorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class LogarithmPCACalculatorConfig(BaseCalculatorConfig):
    """Configuration class for PCA Calculator.

    Attributes:
        n_components (int): The number of principal components to compute. Defaults to 1.
        pca_default_weights (Optional[List[float]]): Initial weights for PCA calculation. Defaults to None.
        var_normalized (bool): Flag to determine if variable normalization is applied. Defaults to False.
        logarithm_transform (bool): Flag to determine if logarithm transformation is applied. Defaults to True.
        logarithm_smoothing_term (float): Smoothing term for logarithm transformation. Defaults to 1e-8.
    """

    n_components: int = 1
    pca_default_weights: Optional[List[float]] = None
    var_normalized: bool = False
    logarithm_transform: bool = True
    logarithm_smoothing_term: float = 1e-8


class LogarithmPCACalculator(BaseCalculator):
    """PCA Calculator for computing principal components analysis.

    Args:
        dataframe (pd.DataFrame): The input data frame containing the data to be analyzed.
        config (Optional[Dict]): Configuration dictionary to initialize PCACalculatorConfig. Defaults to None.

    Attributes:
        config (PCACalculatorConfig): Configuration object for PCA calculation.
        n_components (int): Number of principal components to compute.
        pca_default_weights (np.ndarray): Weights for PCA calculation.
        var_normalized (bool): Indicates if the variables are normalized.
    """

    def __init__(self, dataframe: pd.DataFrame, config: Optional[Dict] = None) -> None:
        super().__init__(dataframe, config)
        self.config = LogarithmPCACalculatorConfig(**(config or {}))
        self.n_components = self.config.n_components
        if self.config.pca_default_weights is None:
            self.pca_default_weights = None
        else:
            self.pca_default_weights = np.asarray(self.config.pca_default_weights)
        self.var_normalized = self.config.var_normalized
        self.logarithm_transform = self.config.logarithm_transform
        self.logarithm_smoothing_term = self.config.logarithm_smoothing_term
        self._data_preprocessing()
        self.viewer_instance = DistributionViewer(self.dataframe, self.score_columns)
        self.calculate()

    def _data_preprocessing(self) -> None:
        """Preprocesses the input data for PCA calculation."""
        logger.info("Preprocessing data for PCA calculation...")
        self.clean_dataframe = self.dataframe.dropna()
        self.data = self.clean_dataframe[self.score_columns].to_numpy()
        self.data_means = np.mean(self.data, axis=0)
        if self.logarithm_transform:
            logger.info("Applying logarithm transformation to data...")
            data = np.log10(self.logarithm_smoothing_term + self.data)
            self.logarithm_data_means = np.mean(data, axis=0)
            data_centered = data - self.logarithm_data_means
        else:
            data_centered = self.data - np.mean(self.data, axis=0)
        if self.var_normalized:
            logger.info("Normalizing variables in data...")
            data_normalized = data_centered / np.std(data_centered, axis=0)
        else:
            data_normalized = data_centered
        self.data_normalized = data_normalized

    @staticmethod
    def weighted_pca(
        data_normalized: np.ndarray, weights: Optional[np.ndarray], n_components: int
    ) -> tuple:
        """Performs weighted PCA on the provided data.

        Args:
            data (np.ndarray): Input data for PCA.
            weights (Optional[np.ndarray]): Weights for each feature in the data. Defaults to None.
            n_components (int): Number of principal components to compute.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the projected data, eigenvalues, and eigenvectors.
        """
        logger.info("Performing weighted logarithm PCA on the normalized data...")
        if weights is None:
            weights = np.ones(data_normalized.shape[1])
        data_weighted = data_normalized * weights
        cov_matrix = np.cov(data_weighted.T)
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
        idx = eigen_values.argsort()[::-1]
        sorted_eigenvalues = eigen_values[idx]
        sorted_eigenvectors = eigen_vectors[:, idx]
        sorted_eigenvalues = sorted_eigenvalues[:n_components]
        sorted_eigenvectors = sorted_eigenvectors[:, :n_components]
        projected_data = np.dot(data_normalized, sorted_eigenvectors)
        return projected_data, sorted_eigenvalues, sorted_eigenvectors

    def calculate(self) -> None:
        """Calculates the PCA based on the initialized configuration and updates the instance attributes with the results."""
        logger.info("Calculating weighted PCA...")
        self.projected_data, self.sorted_eigenvalues, self.sorted_eigenvectors = (
            self.weighted_pca(
                self.data_normalized,
                self.pca_default_weights,
                self.n_components,
            )
        )

    def update_pca_weights(self, pca_weights: np.ndarray) -> None:
        """Updates the PCA weights and recalculates the PCA.

        Args:
            pca_weights (np.ndarray): New PCA weights to be applied.
        """
        self.pca_default_weights = pca_weights
        self.calculate()

    def plot_logarithm_distributions(self) -> None:
        """Plots the logarithmic distribution of the scores in the dataframe."""
        self.viewer_instance.plot_logarithm_distributions()

    def plot_projected_distribution(self) -> None:
        """Plots the projected data after Logarithm PCA."""
        self.viewer_instance.plot_array_distribution(
            scores=self.projected_data,
            legend="Projected Data",
        )
