from __future__ import annotations

import numpy as np
import pytest
from pytest_cases import case, parametrize_with_cases

from mfpbench.metric import Metric, OutOfBoundsError


@case
def case_metric_minimize_unbounded() -> tuple[Metric, float, float]:
    metric = Metric("m", minimize=True)
    return (metric, 0.5, 0.5)


@case
def case_metric_minimize_lower_bounded() -> tuple[Metric, float, float]:
    metric = Metric("m", minimize=True, bounds=(-1, np.inf))
    return (metric, 0.5, 1.5)


@case
def case_metric_minimize_upper_bounded() -> tuple[Metric, float, float]:
    metric = Metric("m", minimize=True, bounds=(-np.inf, 1))
    return (metric, 0.5, 0.5)


@case
def case_metric_minimize_bounded() -> tuple[Metric, float, float]:
    metric = Metric("m", minimize=True, bounds=(-1, 1))
    return (metric, 0.5, 0.75)


@case
def case_metric_maximize_unbounded() -> tuple[Metric, float, float]:
    metric = Metric("m", minimize=False)
    return (metric, 0.5, -0.5)


@case
def case_metric_maximize_lower_bounded() -> tuple[Metric, float, float]:
    metric = Metric("m", minimize=False, bounds=(-1, np.inf))
    return (metric, 0.5, -0.5)


@case
def case_metric_maximize_upper_bounded() -> tuple[Metric, float, float]:
    metric = Metric("m", minimize=False, bounds=(-np.inf, 1))
    return (metric, 0.25, 0.75)


@case
def case_metric_maximize_bounded() -> tuple[Metric, float, float]:
    metric = Metric("m", minimize=False, bounds=(-1, 1))
    return (metric, 0, 0.5)


@parametrize_with_cases("metric, value, expected", cases=".")
def test_metric_cost(metric: Metric, value: float, expected: float) -> None:
    assert metric.as_minimize_value(value) == expected


def test_metric_complains_if_out_of_bounds() -> None:
    metric = Metric("m", minimize=True, bounds=(-1, 1))
    with pytest.raises(OutOfBoundsError):
        metric.as_minimize_value(-2)
    with pytest.raises(OutOfBoundsError):
        metric.as_minimize_value(2)
