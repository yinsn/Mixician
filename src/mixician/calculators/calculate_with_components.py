from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseCalculator, BaseCalculatorConfig


class PCACalculatorConfig(BaseCalculatorConfig):
    """Configuration for PCA Calculator.

    Attributes:
        n_components (int): Number of principal components to use.
    """

    n_components: int = 1
    pca_default_weights: Optional[np.ndarray] = None


class PCACalculator(BaseCalculator):

    def __init__(self, dataframe: pd.DataFrame, config: Optional[Dict] = None) -> None:
        super().__init__(dataframe, config)
        self.config = PCACalculatorConfig(**(config or {}))
        self.n_components = self.config.n_components
        self.pca_default_weights = self.config.pca_default_weights
