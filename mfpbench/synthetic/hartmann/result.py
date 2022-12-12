from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from mfpbench.result import Result
from mfpbench.synthetic.hartmann.config import MFHartmannConfig

C = TypeVar("C", bound=MFHartmannConfig)


@dataclass(frozen=True)  # type: ignore[misc]
class MFHartmannResult(Result[C, int]):
    value: float
    fid_cost: float

    @property
    def score(self) -> float:
        """The score of interest."""
        # TODO: what should be an appropriate score since flipping signs may not be
        #  adequate or meaningful. When is the property score used?
        # Hartmann functions have multiple minimas with the global valued at < 0
        # The function evaluates to a y-value that needs to be minimized
        #  https://www.sfu.ca/~ssurjano/hart3.html
        return self.value

    @property
    def error(self) -> float:
        """The score of interest."""
        # TODO: verify
        # Hartmann functions have multiple minimas with the global valued at < 0
        # The function evaluates to a y-value that needs to be minimized
        #  https://www.sfu.ca/~ssurjano/hart3.html
        return self.value

    @property
    def test_score(self) -> float:
        """Just returns the score."""
        return self.score

    @property
    def test_error(self) -> float:
        """Just returns the error."""
        return self.error

    @property
    def val_score(self) -> float:
        """Just returns the score."""
        return self.score

    @property
    def val_error(self) -> float:
        """Just returns the error."""
        return self.error

    @property
    def cost(self) -> float:
        """Just retuns the fidelity."""
        # return self.fidelity
        return self.fid_cost
