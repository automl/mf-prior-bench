from __future__ import annotations

from itertools import combinations

import numpy as np
import scipy.stats
from numpy.linalg import norm
from typing_extensions import Literal


def spearmanr(X: np.ndarray) -> np.ndarray:
    corr, _ = scipy.stats.spearmanr(X, axis=1)
    if isinstance(corr, float):
        result = np.eye(N=2, dtype=float)
        result[0, 1] = corr
        result[1, 0] = corr
    else:
        result = corr

    return result


def kendalltau(X: np.ndarray) -> np.ndarray:
    results = np.eye(N=len(X), dtype=float)

    idxs = range(len(X))
    for i, j in combinations(idxs, 2):
        x = X[i, :]
        y = X[j, :]

        corr, _ = scipy.stats.kendalltau(x, y)

        results[i, j] = corr
        results[j, i] = corr

    return results


def cosine(X: np.ndarray) -> np.ndarray:
    X_norm = np.zeros(shape=X.shape, like=X)
    for i, row in enumerate(X):
        mi = row.min()
        ma = row.max()
        X_norm[i] = 2 * ((row - mi) / (ma - mi)) - 1

    results = np.eye(N=len(X), dtype=float)
    idxs = range(len(X))
    for i, j in combinations(idxs, 2):
        x = X_norm[i, :10]
        y = X_norm[j, :10]

        corr = np.dot(x, y) / (norm(x) * norm(y))

        results[i, j] = corr
        results[j, i] = corr

    return results


def rank_correlation(
    x: np.ndarray,
    *,
    method: Literal["spearman", "kendalltau", "cosine"] = "cosine",
) -> np.ndarray:
    """Calculate rank correlation between observer rankings.

    Will return the correlation between the two rankings,
    otherwise it will return a correlation matrix where each row represents a random
    variable.

    Parameters
    ----------
    x: np.ndarray
        The rankings to calculate correlations for where each row
        represents a random variable.

    method: Literal["spearman", "kendalltau", "cosine"] = "cosine"
        The method to use

    Returns
    -------
    np.ndarray
        The correlation matrix
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    assert x.ndim >= 2

    measures = {"spearman": spearmanr, "kendalltau": kendalltau, "cosine": cosine}
    f = measures.get(method)
    if f is None:
        raise NotImplementedError(method)

    return f(x)
