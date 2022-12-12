from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Generic, Mapping, TypeVar

from mfpbench.config import Config

# Used to get correct and good typing
SelfT = TypeVar("SelfT", bound="Result")

# The Config kind
C = TypeVar("C", bound=Config)

# Fidelity kind
F = TypeVar("F", int, float)


@dataclass(frozen=True)  # type: ignore[misc]
class Result(ABC, Generic[C, F]):
    """Collect all results in a class for clarity."""

    config: C
    fidelity: F

    @classmethod
    def from_dict(
        cls: type[SelfT],
        config: C,
        result: Mapping[str, Any],
        fidelity: F,
    ) -> SelfT:
        # To help with serialization, we need to convert floats to... ehh floats
        # This is due to some things returning an np.float -_-
        result = {k: float(v) if isinstance(v, float) else v for k, v in result.items()}
        return cls(config=config, fidelity=fidelity, **result)

    @property
    @abstractmethod
    def score(self) -> float:
        """The score of interest."""
        ...

    @property
    @abstractmethod
    def error(self) -> float:
        """The error of interest."""
        ...

    @property
    @abstractmethod
    def test_score(self) -> float:
        """The score on the test set."""
        ...

    @property
    @abstractmethod
    def test_error(self) -> float:
        """The error on the test set."""
        ...

    @property
    @abstractmethod
    def val_score(self) -> float:
        """The score on the validation set."""
        ...

    @property
    @abstractmethod
    def val_error(self) -> float:
        """The score on the validation set."""
        ...

    @property
    @abstractmethod
    def cost(self) -> float:
        """The time cost for evaluting this config."""
        ...

    def dict(self) -> dict[str, Any]:
        """Create a dict from this result."""
        d = asdict(self)
        del d["config"]
        del d["fidelity"]
        return d
