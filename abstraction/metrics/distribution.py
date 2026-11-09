import numpy as np
from scipy import linalg


def frechet_distance(a, b):
    """Frechet distance between Gaussians fitted to two feature sets (rows are samples).

    With DINOv2 features this is FD-DINOv2. The estimate is biased upward for
    small samples; both sample sizes have to be reported with the value.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(cov_a @ cov_b, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_a.shape[0]) * 1e-6
        covmean, _ = linalg.sqrtm((cov_a + offset) @ (cov_b + offset), disp=False)
    covmean = covmean.real
    diff = mu_a - mu_b
    return float(diff @ diff + np.trace(cov_a) + np.trace(cov_b) - 2 * np.trace(covmean))


def kernel_distance(a, b, subset_size=1000, n_subsets=100, seed=0):
    """Kernel distance (KID): unbiased MMD^2 with the kernel (x.y / d + 1)^3.

    Returns (mean, standard deviation) over random subsets.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a.shape[1]
    m = min(subset_size, len(a), len(b))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_subsets):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        values.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                      + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                      - 2 * kxy.mean())
    return float(np.mean(values)), float(np.std(values))
