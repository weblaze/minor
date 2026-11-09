import numpy as np


def participation_ratio(x):
    """Effective dimensionality (sum of eigenvalues)^2 / sum of squared eigenvalues of the covariance."""
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    x = x - x.mean(0)
    s = np.linalg.svd(x, compute_uv=False) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum())
