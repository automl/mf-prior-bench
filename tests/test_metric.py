from __future__ import annotations

import numpy as np
import pytest
from pytest_cases import case, parametrize_with_cases

from mfpbench.metric import Metric, OutOfBoundsError

# NOTE: Each case returns the Metric, the value to use and a tuple of (score, error)


# MINIMIZE
@case
def case_metric_minimize_unbounded() -> tuple[Metric, float, tuple[float, float]]:
    metric = Metric(minimize=True)
    return metric, 0.5, (-0.5, 0.5)


@case
def case_metric_minimize_lower_bounded() -> tuple[Metric, float, tuple[float, float]]:
    metric = Metric(minimize=True, bounds=(-1, np.inf))
    return metric, 0.5, (-0.5, 0.5)


@case
def case_metric_minimize_upper_bounded() -> tuple[Metric, float, tuple[float, float]]:
    metric = Metric(minimize=True, bounds=(-np.inf, 1))
    return metric, 0.5, (-0.5, 0.5)


@case
def case_metric_minimize_bounded() -> tuple[Metric, float, tuple[float, float]]:
    metric = Metric(minimize=True, bounds=(-1, 1))
    return metric, 0.5, (0.25, 0.75)


# MAXIMIZE
@case
def case_metric_maximize_unbounded() -> tuple[Metric, float, tuple[float, float]]:
    metric = Metric(minimize=False)
    return metric, 0.5, (0.5, -0.5)


@case
def case_metric_maximize_lower_bounded() -> tuple[Metric, float, tuple[float, float]]:
    metric = Metric(minimize=False, bounds=(-1, np.inf))
    return metric, 0.5, (0.5, -0.5)


@case
def case_metric_maximize_upper_bounded() -> tuple[Metric, float, tuple[float, float]]:
    metric = Metric(minimize=False, bounds=(-np.inf, 1))
    return (metric, 0.25, (0.25, -0.25))


@case
def case_metric_maximize_bounded() -> tuple[Metric, float, tuple[float, float]]:
    metric = Metric(minimize=False, bounds=(-1, 1))
    return (metric, 0.5, (0.75, 0.25))


@parametrize_with_cases("metric, value, expected", cases=".")
def test_metric_error(
    metric: Metric,
    value: float,
    expected: tuple[float, float],
) -> None:
    _, error = expected
    assert metric.as_value(value).error == error


@parametrize_with_cases("metric, value, expected", cases=".")
def test_metric_score(
    metric: Metric,
    value: float,
    expected: tuple[float, float],
) -> None:
    score, _ = expected
    assert metric.as_value(value).score == score


@parametrize_with_cases("metric, value, expected", cases=".")
def test_metric_value(
    metric: Metric,
    value: float,
    expected: tuple[float, float],  # noqa: ARG001
) -> None:
    assert metric.as_value(value).value == value


def test_metric_complains_if_out_of_bounds() -> None:
    metric = Metric(minimize=True, bounds=(-1, 1))
    with pytest.raises(OutOfBoundsError):
        metric.as_value(-2)
    with pytest.raises(OutOfBoundsError):
        metric.as_value(2)
