from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Generic, Mapping, TypeVar
from typing_extensions import Self, override

from mfpbench.config import Config

# The Config kind
C = TypeVar("C", bound=Config)

# Fidelity kind
F = TypeVar("F", int, float)


@dataclass(frozen=True)  # type: ignore[misc]
class Result(ABC, Generic[C, F]):
    """Collect all results in a class for clarity."""

    fidelity: F
    """The fidelity of this result."""

    config: C = field(repr=False)
    """The config used to generate this result."""

    @classmethod
    def from_dict(
        cls,
        config: C,
        result: Mapping[str, Any],
        fidelity: F,
    ) -> Self:
        """Create from a dict or mapping object."""
        fieldnames = set(cls.names())
        if not fieldnames.issubset(result.keys()):
            raise ValueError(
                f"Result dict is missing fields: {fieldnames - result.keys()}",
            )
        # To help with serialization, we need to convert floats to... ehh floats
        # This is due to some things returning an np.float -_-
        result = {
            k: float(v) if isinstance(v, float) else v
            for k, v in result.items()
            if k in fieldnames
        }
        return cls(config=config, fidelity=fidelity, **result)

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """The names of the fields in this result."""
        return tuple(
            f.name for f in fields(cls) if f.name not in ("config", "fidelity")
        )

    @classmethod
    def from_row(
        cls,
        config: C,
        row: Mapping[str, Any],
        fidelity: F,
    ) -> Self:
        """Create from a row of a dataframe."""
        return cls.from_dict(config, dict(row), fidelity)

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


@dataclass(frozen=True, eq=False)  # type: ignore[misc]
class GenericTabularResult(Result[C, F], Generic[C, F]):
    """A generic tabular result.

    This is useful for adhoc tabular benchmarks.
    """

    _values: dict[str, Any]

    def __hash__(self) -> int:
        """Hash based on the dictionary repr."""
        return (
            hash(self.config) ^ hash(self.fidelity) ^ hash(tuple(self._values.items()))
        )

    def dict(self) -> Any:
        """As a raw dictionary."""
        return dict(self._values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    # Make .property acces work
    def __getattr__(self, __name: str) -> Any:
        return self._values[__name]

    @override
    @classmethod
    def from_dict(cls, config: C, result: Mapping[str, Any], fidelity: F) -> Self:
        """Create from a dict or mapping object."""
        return cls(config=config, _values=dict(result), fidelity=fidelity)

    @property
    def score(self) -> float:
        """The score of interest."""
        if "score" in self._values:
            return float(self._values["score"])

        raise KeyError("GenericTabularResult does not have a score")

    @property
    def error(self) -> float:
        """The error of interest."""
        if "error" in self._values:
            return float(self._values["error"])

        raise KeyError("GenericTabularResult does not have an error")

    @property
    def test_score(self) -> float:
        """The score on the test set."""
        if "test_score" in self._values:
            return float(self._values["test_score"])

        raise KeyError("GenericTabularResult does not have a test_score")

    @property
    def test_error(self) -> float:
        """The error on the test set."""
        if "test_error" in self._values:
            return float(self._values["test_error"])

        raise KeyError("GenericTabularResult does not have a test_error")

    @property
    def val_score(self) -> float:
        """The score on the validation set."""
        if "val_score" in self._values:
            return float(self._values["val_score"])

        raise KeyError("GenericTabularResult does not have a val_score")

    @property
    def val_error(self) -> float:
        """The score on the validation set."""
        if "val_error" in self._values:
            return float(self._values["val_error"])

        raise KeyError("GenericTabularResult does not have a val_error")

    @property
    def cost(self) -> float:
        """The time cost for evaluting this config."""
        if "cost" in self._values:
            return float(self._values["cost"])

        raise KeyError("GenericTabularResult does not have a cost")

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """The names of the fields in this result."""
        return tuple(
            f.name
            for f in fields(cls)
            if f.name not in ("config", "fidelity", "__values")
        )
