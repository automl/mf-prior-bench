from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from dataclasses import asdict, dataclass

# Just so `def copy(...)` can give back the correct type
SelfT = TypeVar("SelfT", bound="Config")


@dataclass(frozen=True)  # type: ignore[misc]
class Config(ABC):
    """A Config used to query a benchmark

    * Include all hyperparams
    * Includes the fidelity
    * Configuration and generation agnostic
    * Immutable to prevent accidental changes during running, mutate with copy and
        provide arguments as required.
    * Easy equality between configs
    """

    def dict(self) -> dict[str, Any]:
        """Converts the config to a raw dictionary"""
        return asdict(self)

    @classmethod
    def mutate(cls: type[SelfT], original: SelfT, **kwargs: Any) -> SelfT:
        """Copy a config and mutate it if needed"""
        d = asdict(original)
        d.update(kwargs)
        return cls(**d)

    def copy(self: SelfT, **kwargs: Any) -> SelfT:
        """Copy this config and mutate it if needed"""
        return self.mutate(self, **kwargs)

    @abstractmethod
    def validate(self) -> None:
        """Validate the config, just useful early on while testing

        Raises
        ------
        AssertionError
        """
        ...
