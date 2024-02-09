import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate the Kullback-Leibler divergence between two probability distributions.

    Args:
        p (np.ndarray): The first probability distribution as a numpy array.
        q (np.ndarray): The second probability distribution as a numpy array.

    Returns:
        float: The Kullback-Leibler divergence between distributions p and q.
    """
    return float(np.sum(np.where(p != 0, p * np.log(p / q), 0)))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate the Jensen-Shannon divergence between two probability distributions.

    This is a symmetric measure derived from the Kullback-Leibler divergence, and
    it measures the similarity between two probability distributions. It's the
    square root of the Jensen-Shannon distance.

    Args:
        p (np.ndarray): The first probability distribution as a numpy array.
        q (np.ndarray): The second probability distribution as a numpy array.

    Returns:
        float: The Jensen-Shannon divergence between distributions p and q.
    """
    m = 0.5 * (p + q)
    return float(np.sqrt(0.5 * (kl_divergence(p, m) + kl_divergence(q, m))))
