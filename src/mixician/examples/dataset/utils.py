import logging
import os
import random
from typing import Optional

import numpy as np
import pandas as pd

from ...dataloaders import ensure_study_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_lognormal_data_near_zero(
    size: tuple[int, int], mean: float, sigma: float
) -> np.ndarray:
    """
    Generates lognormal data near zero, clipped to the range [0, 1].

    Args:
        size (Tuple[int, int]): The shape of the generated data array (rows, columns).
        mean (float): The mean value for the lognormal distribution.
        sigma (float): The standard deviation for the lognormal distribution.

    Returns:
        np.ndarray: An array of lognormal data clipped to [0, 1].
    """
    data = np.asarray(np.random.lognormal(mean, sigma, size) / np.log(10))
    return np.clip(data, 0, 1)


def create_random_boolean_labels(
    dataframe: pd.DataFrame,
    reference_column: str,
    rows: int,
) -> np.ndarray:
    """
    Creates random boolean labels for the specified number of rows.
    """
    labels = np.zeros(rows, dtype=int)
    top_indices = dataframe[reference_column] > dataframe[reference_column].quantile(
        0.95
    )

    labels[top_indices] = 1
    return labels


def create_random_numeric_labels(
    dataframe: pd.DataFrame,
    reference_column: str,
    rows: int,
    seed: int = 42,
    low: float = 0.0,
    high: float = 10000.0,
) -> np.ndarray:
    """
    Creates random numeric labels for the specified number of rows.
    """
    random.seed(seed)
    labels = np.zeros(rows, dtype=float)
    num_non_zero = int(0.1 * rows)
    non_zero_indices = np.random.choice(
        np.arange(rows), size=num_non_zero, replace=False
    )
    sorted_indices = non_zero_indices[
        np.argsort(dataframe.iloc[non_zero_indices][reference_column])
    ]
    min_value, max_value = low, high
    sorted_random_values = np.linspace(min_value, max_value, num_non_zero)
    labels[sorted_indices] = sorted_random_values
    return labels


def create_mix_rank_test_samples(
    rows: int,
    num_page_types: int,
    num_features: int,
    num_boolean_labels: int,
    num_numeric_labels: int,
    lognormal_mean_lower_bound: float = -6,
    lognormal_sigma_upper_bound: float = 1,
    seed: int = 42,
    study_path: Optional[str] = None,
    study_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Creates mixed rank test samples with specified characteristics and saves to a CSV file.

    Args:
        rows (int): Number of rows (samples) to generate.
        num_page_types (int): Number of distinct page types/categories.
        num_features (int): Number of features to generate per sample.
        num_boolean_labels (int): Number of boolean labels to generate per sample.
        num_numeric_labels (int): Number of numeric labels to generate per sample.
        lognormal_mean_lower_bound (float, optional): Lower bound for the mean of the lognormal distribution. Defaults to -6.
        lognormal_sigma_upper_bound (float, optional): Upper bound for the sigma of the lognormal distribution. Defaults to 1.
        seed (int, optional): Seed for the random number generator. Defaults to 42.
        study_path (Optional[str], optional): Base path where the study directory will be created or ensured. Defaults to None.
        study_name (Optional[str], optional): Name of the study, used to create or ensure the study directory. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the generated test samples.
    """
    logger.info(f"Generating mixed rank test samples for {rows} rows...")

    random.seed(seed)
    np.random.seed(seed)

    request_ids = [random.randint(1, rows // 100) for _ in range(rows)]
    item_ids = [random.randint(1, rows) for _ in range(rows)]
    page_types = [f"page_type_{i+1}" for i in range(num_page_types)]
    categories = [random.choice(page_types) for _ in range(rows)]

    df = pd.DataFrame(
        {"request_id": request_ids, "item_id": item_ids, "category": categories}
    )

    lognormal_mean = np.random.uniform(lognormal_mean_lower_bound, 0, num_features)
    lognormal_sigma = np.random.uniform(0, lognormal_sigma_upper_bound, num_features)

    for i in range(num_features):
        lognormal_data = generate_lognormal_data_near_zero(
            size=(rows, 1),
            mean=lognormal_mean[i],
            sigma=lognormal_sigma[i],
        )
        df[f"pxtr_{i+1}"] = lognormal_data

    df = df.sort_values(by="request_id").reset_index(drop=True)
    feature_columns = [f"pxtr_{i+1}" for i in range(num_features)]
    df["total_score"] = df[feature_columns].prod(axis=1)

    for i in range(num_boolean_labels):
        df[f"boolean_label_{i+1}"] = create_random_boolean_labels(
            dataframe=df,
            reference_column="total_score",
            rows=rows,
        )

    for i in range(num_numeric_labels):
        df[f"numeric_label_{i+1}"] = create_random_numeric_labels(
            dataframe=df,
            reference_column="total_score",
            rows=rows,
            seed=seed + i,
        )
    df = df.drop(columns=["total_score"])

    study_path = ensure_study_directory(
        study_path=study_path,
        study_name=study_name,
    )

    export_path = os.path.join(study_path, "mix_rank_test_samples.csv")
    df.to_csv(export_path, index=False)

    return df
