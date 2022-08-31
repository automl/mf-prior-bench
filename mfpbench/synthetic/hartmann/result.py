from __future__ import annotations

from typing import Any, Mapping, TypeVar

from dataclasses import dataclass

from mfpbench.result import Result
from mfpbench.synthetic.hartmann.config import MFHartmannConfig

C = TypeVar("C", bound=MFHartmannConfig)


@dataclass(frozen=True)
class MFHartmannResult(Result[C, int]):
    z: int
    value: float

    @classmethod
    def from_dict(
        cls,
        config: C,
        result: Mapping[str, Any],
        fidelity: int,
    ) -> MFHartmannResult:
        """Create a MFHartmannResult from a dictionary

        Parameters
        ----------
        config : MFHartmannConfig
            The config the result is from

        result : Mapping[str, Any]
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
    def fidelity(self) -> int:
        """The fidelity this result is from"""
        return self.z

    @property
    def cost(self) -> float:
        """Just retuns the fidelity"""
        return self.z
