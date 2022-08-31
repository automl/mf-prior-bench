from __future__ import annotations

from typing import TypeVar

from dataclasses import dataclass

from mfpbench.result import Result
from mfpbench.synthetic.hartmann.config import MFHartmannConfig

C = TypeVar("C", bound=MFHartmannConfig)


@dataclass(frozen=True)  # type: ignore[misc]
class MFHartmannResult(Result[C, int]):
    value: float

    @property
    def score(self) -> float:
        """The score of interest"""
        return self.value

    @property
    def error(self) -> float:
        """The score of interest"""
        return -self.value

    @property
    def test_score(self) -> float:
        """Just returns the score"""
        return self.score

    @property
    def test_error(self) -> float:
        """Just returns the error"""
        return self.error

    @property
    def val_score(self) -> float:
        """Just returns the score"""
        return self.score

    @property
    def val_error(self) -> float:
        """Just returns the error"""
        return self.error

    @property
    def cost(self) -> float:
        """Just retuns the fidelity"""
        return self.fidelity
