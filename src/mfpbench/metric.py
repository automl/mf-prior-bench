from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class OutOfBoundsError(ValueError):
    """Raised when a value is outside of the bounds of a metric."""


@dataclass
class Metric:
    """A metric to be used in the benchmark.

    It's main use is to convert a raw value into a value that can always be
    minimized.
    """

    name: str
    """The name of the metric."""

    minimize: bool
    """Whether or not to minimize the metric."""

    bounds: tuple[float, float] = field(default_factory=lambda: (-np.inf, np.inf))
    """The bounds of the metric."""

    def __post_init__(self) -> None:
        if self.bounds[0] >= self.bounds[1]:
            raise ValueError(
                f"bounds[0] must be less than bounds[1], got {self.bounds}",
            )

    def as_minimize_value(self, value: float) -> float:
        """Calculate a minimization value for the metric based on its raw value.

        The calculation is as follows:

            | direction | lower | upper |     | minimize_value                     |
            |-----------|-------|-------|-----|------------------------------------|
            | minimize  | inf   | inf   |     | value                              |
            | minimize  | A     | inf   |     | abs(A-value) # distance to optimal |
            | minimize  | inf   | B     |     | value                              |
            | minimize  | A     | B     |     | (value - A) / (B - A)  # 0-1 |
            | ---       | ---   | ---   | --- | ---                                |
            | maximize  | inf   | inf   |     | -value  # convert to minimize      |
            | maximize  | A     | inf   |     | -value # convert to minimize       |
            | maximize  | inf   | B     |     | abs(B - value) # distance optimal  |
            | maximize  | A     | B     |     | (value - A) / (B - a) # 0 -1       |

        Returns:
            The cost of the metric.
        """
        lower, upper = self.bounds
        if value < lower or value > upper:
            raise OutOfBoundsError(f"Value {value} is outside of bounds {self.bounds}")

        if self.minimize:
            if np.isinf(lower):
                return value
            if np.isinf(upper):
                return abs(lower - value)

            return (value - lower) / (upper - lower)

        if np.isinf(upper):
            return -value

        if np.isinf(lower):
            return abs(upper - value)

        return (value - lower) / (upper - lower)
