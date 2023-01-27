from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from mfpbench.pd1.config import PD1Config
from mfpbench.result import Result


@dataclass(frozen=True)  # type: ignore[misc]
class PD1Result(Result[PD1Config, int]):
    valid_error_rate: float  # (0, 1)
    train_cost: float  #

    @property
    def score(self) -> float:
        """The score of interest."""
        return 1 - self.valid_error_rate

    @property
    def error(self) -> float:
        """The error of interest."""
        return self.valid_error_rate

    @property
    def val_score(self) -> float:
        """The score on the validation set."""
        return 1 - self.valid_error_rate

    @property
    def val_error(self) -> float:
        """The error on the validation set."""
        return self.valid_error_rate

    @property
    def cost(self) -> float:
        warnings.warn(f"Unsure of unit for `cost` on {self.__class__}")
        return self.train_cost


@dataclass(frozen=True)  # type: ignore[misc]
class PD1ResultSimple(PD1Result):
    """Used for all PD1 benchmarks, except imagenet, lm1b, translate_wmt, uniref50."""

    test_error_rate: float = np.inf

    @property
    def test_score(self) -> float:
        """The score on the test set."""
        return self.test_error_rate

    @property
    def test_error(self) -> float:
        """The error on the test set."""
        return 1 - self.test_error_rate


@dataclass(frozen=True)
class PD1ResultTransformer(PD1Result):
    """Imagenet, lm1b, translate_wmt, uniref50, cifar100 contains no test error."""

    @property
    def test_score(self) -> float:
        """The score on the test set."""
        warnings.warn("Using valid error rate as there is no test error rate")
        return self.valid_error_rate

    @property
    def test_error(self) -> float:
        """The error on the test set."""
        warnings.warn("Using valid error rate as there is no test error rate")
        return self.valid_error_rate
