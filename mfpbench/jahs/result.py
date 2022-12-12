from __future__ import annotations

import warnings
from dataclasses import dataclass

from mfpbench.jahs.config import JAHSConfig
from mfpbench.result import Result


@dataclass(frozen=True)  # type: ignore[misc]
class JAHSResult(Result[JAHSConfig, int]):
    # Info
    # size: float  # remove
    # flops: float # remove
    # latency: float  # unit? remove
    runtime: float  # unit?

    # Scores (0 - 100)
    valid_acc: float
    test_acc: float
    # train_acc: float # remove

    @property
    def score(self) -> float:
        """The score of interest."""
        return self.valid_acc

    @property
    def error(self) -> float:
        """The error of interest."""
        return 100 - self.valid_acc

    @property
    def test_score(self) -> float:
        """The score on the test set."""
        return self.test_acc

    @property
    def test_error(self) -> float:
        """The error on the test set."""
        return 100 - self.test_acc

    @property
    def val_score(self) -> float:
        """The score on the validation set."""
        return self.valid_acc

    @property
    def val_error(self) -> float:
        """The error on the validation set."""
        return 100 - self.valid_acc

    @property
    def cost(self) -> float:
        """The time taken in seconds."""
        warnings.warn(f"Unsure of unit for `cost` on {self.__class__}")
        return self.runtime
