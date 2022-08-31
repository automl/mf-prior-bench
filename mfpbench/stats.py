from __future__ import annotations

from typing import Sequence, overload

import numpy as np
from scipy.stats import spearmanr
from typing_extensions import Literal


@overload
def rank_correlation(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    method: Literal["spearman"],
) -> float:
    ...


@overload
def rank_correlation(
    x: Sequence[float] | np.ndarray,
    y: None = None,
    *,
    method: Literal["spearman"],
) -> np.ndarray:
    ...


def rank_correlation(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray | None = None,
    *,
    method: Literal["spearman"] = "spearman",
) -> float | np.ndarray:
    """Calculate rank correlation between observer rankings

    If both x and y are specified, will return the correlation between the two rankings,
    otherwise it will return a correlation matrix where each row represents a random
    variable.

    Parameters
    ----------
    x: Sequence[float] | np.ndarray
    y: Sequence[float] | np.ndarray | None = None
    method: Literal["spearman"] = "spearman"

    Returns
    -------
    float | np.ndarray
        The correlation or correlation matrix
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if y is None:
        assert x.ndim >= 2

    if method == "spearman":
        if y is None:
            return spearmanr(x, axis=1).correlation
        else:
            return spearmanr(x, y).correlation
    else:
        raise NotImplementedError(f"Not a supported method {method}")
