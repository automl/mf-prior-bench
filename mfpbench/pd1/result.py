from __future__ import annotations

import warnings
from dataclasses import dataclass

from mfpbench.pd1.config import PD1Config
from mfpbench.result import Result


@dataclass(frozen=True)  # type: ignore[misc]
class PD1Result(Result[PD1Config, int]):

    result1: float
    result2: float

    train_cost: float

    @property
    def score(self) -> float:
        """The score of interest"""
        return self.result1

    @property
    def error(self) -> float:
        """The error of interest"""
        return 1 - self.result1

    @property
    def test_score(self) -> float:
        """The score on the test set"""
        return self.result2

    @property
    def test_error(self) -> float:
        """The error on the test set"""
        return 1 - self.result2

    @property
    def val_score(self) -> float:
        """The score on the validation set"""
        return self.result1

    @property
    def val_error(self) -> float:
        """The error on the validation set"""
        return 1 - self.result2

    @property
    def cost(self) -> float:
        """The time taken in seconds"""
        warnings.warn(f"Unsure of unit for `cost` on {self.__class__}")
        return self.train_cost
