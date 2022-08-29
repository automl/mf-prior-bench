from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping

from mfpbench.jahs.config import JAHSConfig
from mfpbench.result import Result


@dataclass  # type: ignore[misc]
class JAHSResult(Result):
    epoch: int

    # Info
    size: float  # MB
    flops: float
    latency: float  # unit?
    runtime: float  # unit?

    # Scores
    valid_acc: float
    test_acc: float
    train_acc: float

    @classmethod
    def from_dict(
        cls,
        config: JAHSConfig,
        result: Mapping[str, Any],
        fidelity: int,
    ) -> JAHSResult:
        """

        Parameters
        ----------
        config: JAHSConfig
            The config used to generate these results

        result : Mapping[str, Any]
            The results to pull from

        fidelity : int
            The fidelity at which this config was evaluated, epochs

        Returns
        -------
        JAHSResult
        """
        return JAHSResult(
            config=config,
            epoch=fidelity,
            size=result["size_MB"],
            flops=result["FLOPS"],
            latency=result["latency"],
            runtime=result["runtime"],
            valid_acc=result["valid-acc"],
            test_acc=result["test-acc"],
            train_acc=result["train-acc"],
        )

    @property
    def score(self) -> float:
        """The score of interest"""
        return self.valid_acc

    @property
    def error(self) -> float:
        """The error of interest"""
        return 1 - self.valid_acc

    @property
    def test_score(self) -> float:
        """The score on the test set"""
        return self.test_acc

    @property
    def val_score(self) -> float:
        """The score on the validation set"""
        return self.valid_acc

    @property
    def fidelity(self) -> int:
        """The fidelity used"""
        return self.epoch

    @property
    def train_time(self) -> float:
        """The time taken in seconds"""
        warnings.warn(f"Unsure of unit for `train_time` on {self.__class__}")
        return self.runtime
