from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, TypeVar

import numpy as np
import yaml
from ConfigSpace import (
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Constant,
)

from mfpbench.util import perturb

# Just so `def copy(...)` can give back the correct type
SelfT = TypeVar("SelfT", bound="Config")


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class Config(ABC, Mapping[str, Any]):
    """A Config used to query a benchmark.

    * Include all hyperparams
    * Includes the fidelity
    * Configuration and generation agnostic
    * Immutable to prevent accidental changes during running, mutate with copy and
        provide arguments as required.
    * Easy equality between configs
    """

    @classmethod
    def from_dict(cls: type[SelfT], d: Mapping[str, Any]) -> SelfT:
        """Create from a dict or mapping object."""
        return cls(**d)

    @classmethod
    def from_configuration(cls: type[SelfT], config: Configuration) -> SelfT:
        """Create from a ConfigSpace.Configuration."""
        return cls.from_dict(config)

    def dict(self) -> dict[str, Any]:
        """Converts the config to a raw dictionary."""
        return asdict(self)

    @classmethod
    def mutate(cls: type[SelfT], original: SelfT, **kwargs: Any) -> SelfT:
        """Copy a config and mutate it if needed."""
        d = asdict(original)
        d.update(kwargs)
        return cls(**d)

    def copy(self: SelfT, **kwargs: Any) -> SelfT:
        """Copy this config and mutate it if needed."""
        return self.mutate(self, **kwargs)

    def perturb(
        self: SelfT,
        space: ConfigurationSpace,
        *,
        seed: int | np.random.RandomState | None = None,
        std: float | None = None,
        categorical_swap_chance: float | None = None,
    ) -> SelfT:
        """Perturb this config.

        Add gaussian noise to each hyperparameter. The mean is centered at
        the current config.

        Parameters
        ----------
        space: ConfigurationSpace
            The space to perturb in

        seed: int | np.random.RandomState | None = None
            The seed to use for the perturbation

        std: float | None = None
            A value in [0, 1] representing the fraction of the hyperparameter range
            to use as the std. If None, will use keep the current value

        categorical_swap_chance: float | None = None
            The probability that a categorical hyperparameter will be changed
            If None, will use keep the current value

        Returns
        -------
        config: Config
            The perturbed config
        """
        new_values: dict = {}
        for name, value in self.items():
            hp = space[name]
            if isinstance(hp, CategoricalHyperparameter) and categorical_swap_chance:
                new_value = perturb(value, hp, seed=seed, std=categorical_swap_chance)
            elif not isinstance(hp, CategoricalHyperparameter) and std:
                new_value = perturb(value, hp, seed=seed, std=std)
            else:
                new_value = value

            new_values[name] = new_value

        return self.__class__.from_dict(new_values)

    @abstractmethod
    def validate(self) -> None:
        """Validate the config, just useful early on while testing.

        Raises
        ------
        AssertionError
        """
        ...

    def __eq__(self, that: Any) -> bool:
        """Equality is defined in terms of their dictionary repr."""
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
        """Applies this configuration as a prior on a configspace.

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
        """Load a config from a supported file type.

        Note:
        ----
        Only supports yaml and json for now
        """
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")

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
        """Load a config from a yaml file."""
        with path.open("r") as f:
            d = yaml.safe_load(f)
            return cls.from_dict(d)

    @classmethod
    def from_json(cls: type[SelfT], path: Path) -> SelfT:
        """Load a config from a json file."""
        with path.open("r") as f:
            d = json.load(f)
            return cls.from_dict(d)

    def save(self, path: Path, format: str | None = None) -> None:
        """Save the config.

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
