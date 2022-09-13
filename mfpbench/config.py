from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, TypeVar

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from ConfigSpace import Configuration, ConfigurationSpace

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

    @classmethod
    def from_configuration(cls: type[SelfT], config: Configuration) -> SelfT:
        """Create from a ConfigSpace.Configuration"""
        return cls.from_dict(**config)

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

    def set_as_default_prior(self, configspace: ConfigurationSpace) -> None:
        """Applies this configuration as a prior on a configspace

        Note
        ----
        If there is some renaming that needs to be done, this should be overwritten

        Parameters
        ----------
        configspace: ConfigurationSpace
            The space to apply this config to
        """
        for attr in iter(k for k in self.__annotations__ if not k.startswith("_")):
            hp = configspace[attr]
            value = getattr(self, attr)
            hp.default_value = hp.check_default(value)

    @classmethod
    def from_file(cls: type[SelfT], path: Path) -> SelfT:
        """Load a config from a supported file type

        Note
        ----
        Only supports yaml and json for now
        """
        if path.suffix == "json":
            return cls.from_json(path)
        if path.suffix in ["yaml", "yml"]:
            return cls.from_yaml(path)

        # It has no file suffix, just try both
        try:
            return cls.from_yaml(path)
        except yaml.error.YAMLError:
            pass

        try:
            return cls.from_json(path)
        except json.JSONDecodeError:
            pass

        raise ValueError(f"Path {path} is not valid yaml or json")

    @classmethod
    def from_yaml(cls: type[SelfT], path: Path) -> SelfT:
        """Load a config from a yaml file"""
        with path.open("r") as f:
            d = yaml.safe_load(f)
            return cls(**d)

    @classmethod
    def from_json(cls: type[SelfT], path: Path) -> SelfT:
        """Load a config from a json file"""
        with path.open("r") as f:
            d = json.load(f)
            return cls(**d)
