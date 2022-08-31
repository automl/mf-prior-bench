from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, TypeVar

from dataclasses import asdict, dataclass

from ConfigSpace import Configuration

# Just so `def copy(...)` can give back the correct type
SelfT = TypeVar("SelfT", bound="Config")


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class Config(ABC):
    """A Config used to query a benchmark

    * Include all hyperparams
    * Includes the fidelity
    * Configuration and generation agnostic
    * Immutable to prevent accidental changes during running, mutate with copy and
        provide arguments as required.
    * Easy equality between configs
    """

    @classmethod
    def from_dict(cls: type[SelfT], d: Mapping[str, Any]) -> SelfT:
        """Create from a dict or mapping object"""
        return cls(**d)

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

    def __eq__(self, other: Any) -> bool:
        """Equality is defined in terms of their dictionary repr"""
        if isinstance(other, dict):
            return self.dict() == other
        elif isinstance(other, Configuration):
            return self.dict() == {**other}
        elif isinstance(other, self.__class__):
            return self.dict() == other.dict()
        else:
            return False
