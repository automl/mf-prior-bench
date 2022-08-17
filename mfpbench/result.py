from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from dataclasses import dataclass

from mfpbench.config import Config

# Used to get correct and good typing
SelfT = TypeVar("SelfT", bound="Result")

# The Config kind
C = TypeVar("C", bound=Config)

# Fidelity kind
F = TypeVar("F", int, float)


@dataclass  # type: ignore[misc]
class Result(ABC, Generic[C, F]):
    """Collect all results in a class for clarity"""
    config: C

    @classmethod
    @abstractmethod
    def from_dict(cls: type[SelfT], config: C, result: dict, fidelity: F) -> SelfT:
        """

        Parameters
        ----------
        config : C
            THe config which generated this result

        result : dict
            The results themselves

        fidelity : F
            The fidelity at which this was sampled
        """
        ...

    @property
    @abstractmethod
    def score(self) -> float:
        """The score of interest"""
        ...

    @property
    @abstractmethod
    def fidelity(self) -> F:
        """The fidelity used"""
        ...
