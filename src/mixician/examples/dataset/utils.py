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


def create_mix_rank_test_samples(
    rows: int,
    num_page_types: int,
    num_features: int,
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

    study_path = ensure_study_directory(
        study_path=study_path,
        study_name=study_name,
    )

    export_path = os.path.join(study_path, "mix_rank_test_samples.csv")
    df.to_csv(export_path, index=False)

    return df
