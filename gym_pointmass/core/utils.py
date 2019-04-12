
import numpy as np

def encode_one_hot(n, idx):
    """
    Returns a one-hot encoding of the index `idx` to return an n-long vector.
    >>> encode_one_hot(5, 2)
    array([0., 0., 1., 0., 0.])
    """
    return np.eye(n)[idx]