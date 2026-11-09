import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata


def distance_matrix(x, metric="cosine"):
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    d = cdist(x, x, metric=metric)
    np.fill_diagonal(d, 0.0)
    return d


def _upper(d):
    return d[np.triu_indices_from(d, k=1)]


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def mantel(d_a, d_b, method="pearson", n_permutations=9999, seed=0):
    """Mantel test between two distance matrices over the same items.

    Returns (r, p). p is the one-sided permutation p-value for r > 0, computed
    by permuting the rows and columns of d_b jointly:
    p = (1 + #{r_perm >= r}) / (1 + n_permutations).
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"unknown method {method!r}")
    n = d_a.shape[0]
    iu = np.triu_indices(n, k=1)
    a = d_a[iu]
    if method == "spearman":
        a = rankdata(a)
        d_b = _rank_matrix(d_b)
    r = _corr(a, d_b[iu])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        if _corr(a, d_b[np.ix_(perm, perm)][iu]) >= r:
            count += 1
    return r, (1 + count) / (1 + n_permutations)


def _rank_matrix(d):
    iu = np.triu_indices_from(d, k=1)
    ranked = np.zeros_like(d, dtype=np.float64)
    ranked[iu] = rankdata(d[iu])
    return ranked + ranked.T


def knn_overlap(d_a, d_b, k=10):
    """Mean fraction of shared k nearest neighbours. Returns (overlap, chance level)."""
    n = d_a.shape[0]
    a = d_a + np.diag(np.full(n, np.inf))
    b = d_b + np.diag(np.full(n, np.inf))
    na = np.argsort(a, axis=1)[:, :k]
    nb = np.argsort(b, axis=1)[:, :k]
    shared = [len(set(na[i]) & set(nb[i])) for i in range(n)]
    return float(np.mean(shared) / k), k / (n - 1)


def retrieval_accuracy(queries, gallery, metric="cosine"):
    """Top-1 accuracy of retrieving gallery[i] from queries[i]. Chance is 1 / n."""
    d = cdist(np.asarray(queries, dtype=np.float64), np.asarray(gallery, dtype=np.float64), metric=metric)
    return float((d.argmin(axis=1) == np.arange(len(d))).mean())
