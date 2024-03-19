import numpy as np


def find_top_percentile_value(arr: np.ndarray, percentile: float) -> float:
    """
    Finds the value at the specified percentile in a given numpy array.

    This function calculates the value at the given percentile in a 1D numpy array.
    The array is first sorted, and then the value at the specified percentile is identified
    and returned. Percentile should be given as a float between 0 and 100.

    Args:
        arr (np.ndarray): A 1D numpy array of numeric types.
        percentile (float): The percentile to find the value at, specified as a float
            between 0 and 100.

    Returns:
        float: The value in the array at the specified percentile.
    """
    sorted_arr = np.sort(arr)
    index = int(len(arr) * (percentile / 100.0)) - 1
    index = max(0, min(index, len(arr) - 1))
    return float(sorted_arr[-index - 1])
