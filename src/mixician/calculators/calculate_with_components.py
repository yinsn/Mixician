from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseCalculator, BaseCalculatorConfig


class PCACalculatorConfig(BaseCalculatorConfig):
    """Configuration class for PCA Calculator.

    Attributes:
        n_components (int): The number of principal components to compute. Defaults to 1.
        pca_default_weights (Optional[List[float]]): Initial weights for PCA calculation. Defaults to None.
        var_normalized (bool): Flag to determine if variable normalization is applied. Defaults to True.
    """

    n_components: int = 1
    pca_default_weights: Optional[List[float]] = None
    var_normalized: bool = True


class PCACalculator(BaseCalculator):
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
        self.config = PCACalculatorConfig(**(config or {}))
        self.n_components = self.config.n_components
        self.pca_default_weights = np.asarray(self.config.pca_default_weights)
        self.var_normalized = self.config.var_normalized

    @staticmethod
    def weighted_pca(
        data: np.ndarray,
        weights: Optional[np.ndarray],
        n_components: int,
        var_normalized: bool = True,
    ) -> tuple:
        """Performs weighted PCA on the provided data.

        Args:
            data (np.ndarray): Input data for PCA.
            weights (Optional[np.ndarray]): Weights for each feature in the data. Defaults to None.
            n_components (int): Number of principal components to compute.
            var_normalized (bool): If True, variables will be normalized. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the projected data, eigenvalues, and eigenvectors.
        """
        data_centered = data - np.mean(data, axis=0)
        if var_normalized:
            data_normalized = data_centered / np.std(data_centered, axis=0)
        if weights is None:
            weights = np.ones(data.shape[1])
        data_weighted = data_normalized * weights
        cov_matrix = np.cov(data_weighted.T)
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
        idx = eigen_values.argsort()[::-1]
        sorted_eigenvalues = eigen_values[idx]
        sorted_eigenvectors = eigen_vectors[:, idx]
        sorted_eigenvalues = sorted_eigenvalues[:n_components]
        sorted_eigenvectors = sorted_eigenvectors[:, :n_components]
        projected_data = np.dot(data_normalized, sorted_eigenvectors)
        return projected_data, eigen_values, eigen_vectors

    def calculate(self) -> None:
        """Calculates the PCA based on the initialized configuration and updates the instance attributes with the results."""
        data = self.dataframe[self.score_columns].to_numpy()
        self.projected_data, self.eigen_values, self.eigen_vectors = self.weighted_pca(
            data, self.pca_default_weights, self.n_components, self.var_normalized
        )
