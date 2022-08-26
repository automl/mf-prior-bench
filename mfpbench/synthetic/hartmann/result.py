from __future__ import annotations

from typing import TypeVar

from dataclasses import dataclass

from mfpbench.result import Result
from mfpbench.synthetic.hartmann.config import MFHartmannConfig

C = TypeVar("C", bound=MFHartmannConfig)


@dataclass
class MFHartmannResult(Result[C, int]):
    z: int
    value: float

    @classmethod
    def from_dict(
        cls,
        config: C,
        result: dict,
        fidelity: int,
    ) -> MFHartmannResult:
        """Create a MFHartmannResult from a dictionary

        Parameters
        ----------
        config : MFHartmannConfig
            The config the result is from

        result : dict
            The result dictionary

        fidelity : int
            The fidelity this result is from

        Returns
        -------
        MFHartmannResult
        """
        return MFHartmannResult(config=config, value=result["value"], z=fidelity)

    @property
    def score(self) -> float:
        """The score of interest"""
        return -self.value

    @property
    def error(self) -> float:
        """The score of interest"""
        return self.value

    @property
    def test_score(self) -> float:
        """NA"""
        raise RuntimeError(
            "Hartmann synthetic benchmarks have no train/test/val, use ``result.score``"
        )

    @property
    def val_score(self) -> float:
        """NA"""
        raise RuntimeError(
            "Hartmann synthetic benchmarks have no train/test/val, use ``result.score``"
        )

    @property
    def fidelity(self) -> int:
        """The fidelity this result is from"""
        return self.z

    @property
    def train_time(self) -> float:
        """NA"""
        raise RuntimeError("Hartmann synthetic benchmarks have no training time")
