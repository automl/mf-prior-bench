from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Mapping, TypeVar

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml
from ConfigSpace import Configuration, ConfigurationSpace, Constant

# Just so `def copy(...)` can give back the correct type
SelfT = TypeVar("SelfT", bound="Config")


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class Config(ABC, Mapping[str, Any]):
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
        return cls.from_dict(config)

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

    def __eq__(self, that: Any) -> bool:
        """Equality is defined in terms of their dictionary repr"""
        this = self.dict()
        if isinstance(that, dict):
            that = copy.deepcopy(that)
        elif isinstance(that, Configuration):
            that = {**that}
        elif isinstance(that, self.__class__):
            that = that.dict()
        else:
            return False

        this = {
            k: np.round(v, 10) if isinstance(v, float) else v for k, v in this.items()
        }
        that = {
            k: np.round(v, 10) if isinstance(v, float) else v for k, v in that.items()
        }
        return this == that

    def __getitem__(self, key: str) -> Any:
        return self.dict()[key]

    def __len__(self) -> int:
        return len(self.dict())

    def __iter__(self) -> Iterator[str]:
        return self.dict().__iter__()

    def set_as_default_prior(self, configspace: ConfigurationSpace) -> None:
        """Applies this configuration as a prior on a configspace

        Parameters
        ----------
        configspace: ConfigurationSpace
            The space to apply this config to
        """
        # We convert to dict incase there's any special transformation that happen
        d = self.dict()
        for k, v in d.items():
            hp = configspace[k]
            # https://github.com/automl/ConfigSpace/issues/270
            if isinstance(hp, Constant):
                if hp.default_value != v:
                    raise ValueError(
                        f"Constant {k} must be set to the fixed value"
                        f" {hp.default_value}, not {v}"
                    )
                # No need to do anything here
            else:
                hp.default_value = hp.check_default(v)

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
            return cls.from_dict(d)

    @classmethod
    def from_json(cls: type[SelfT], path: Path) -> SelfT:
        """Load a config from a json file"""
        with path.open("r") as f:
            d = json.load(f)
            return cls.from_dict(d)

    def save(self, path: Path, format: str | None = None) -> None:
        """Save the config

        Parameters
        ----------
        path: Path
            Where to save to. Will infer json or yaml based on filename

        format: str | None = None
            The format to save as. Will use file suffix if not provided
        """
        d = self.dict()
        if format is None:
            if path.suffix == "json":
                format = "json"
            elif path.suffix in ["yaml", "yml"]:
                format = "yaml"
            else:
                format = "yaml"

        if format == "yaml":
            with path.open("w") as f:
                yaml.dump(d, f)
        elif format == "json":
            with path.open("w") as f:
                json.dump(d, f)
        else:
            raise ValueError(f"unkown format `format={format}`")
