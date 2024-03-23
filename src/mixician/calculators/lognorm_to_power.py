import numpy as np


def transform_lognorm_to_power_single_channel(
    data: np.ndarray, upper_bound_3sigma: float
) -> tuple[float, float]:
    """
    Transforms log-normal distributed data to a power law distribution for a single channel,
    calculating the first order weight and power weight.

    Args:
        data (np.ndarray): The input data array, assumed to be log-normally distributed.
        upper_bound_3sigma (float): The upper bound of the data within 3 standard deviations.
    """
    logarithm_data = np.log10(data)
    logarithm_data_mean = np.mean(logarithm_data)
    logarithm_data_std = np.std(logarithm_data)
    first_order_weight = 10.0 ** (3 * logarithm_data_std - logarithm_data_mean)
    power_weight = np.log10(upper_bound_3sigma) / (6 * logarithm_data_std)
    return float(first_order_weight), float(power_weight)
