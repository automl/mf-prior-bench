from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, overload

from ConfigSpace import ConfigurationSpace, Configuration

from mfpbench.config import Config
from mfpbench.result import Result

# The kind of Config to the benchmark
C = TypeVar("C", bound=Config)

# The return value from a config query
R = TypeVar("R", bound=Result)

# The kind of fidelity used in the benchmark
F = TypeVar("F", int, float)


class Benchmark(Generic[C, R, F], ABC):
    """Base class for a Benchmark"""

    fidelity_range: tuple[F, F, F]
    Config: type[C]
    Result: type[R]

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.start: F = self.fidelity_range[0]
        self.end: F = self.fidelity_range[1]
        self.step: F = self.fidelity_range[2]

    @abstractmethod
    def query(self, config: C | dict | Configuration, at: F | None = None) -> R:
        """Submit a query and get a result

        Parameters
        ----------
        query: C | dict | Configuration
            The query to use

        at: F | None = None
            The fidelity at which to query, defaults to None which means *maximum*

        Returns
        -------
        R
            The result of the query
        """
        ...

    @abstractmethod
    def trajectory(
        self,
        config: C,
        *,
        frm: F | None = None,
        to: F | None = None,
        step: F | None = None,
    ) -> list[R]:
        """Get the full trajectory of a configuration

        Parameters
        ----------
        config : C
            The config to query

        frm: F | None = None
            Start of the curve, should default to the start

        to: F | None = None
            End of the curve, should default to the total

        step: F | None = None
            Step size, defaults to ``cls.default_step``

        Returns
        -------
        list[R]
            A list of the results for this config
        """
        ...

    # No number specified, just return one config
    @overload
    def sample(self, n: None = None) -> C:
        ...

    # With a number, return many in a list
    @overload
    def sample(self, n: int) -> list[C]:
        ...

    @abstractmethod
    def sample(self, n: int | None = None) -> C | list[C]:
        """Sample a random possible config

        Parameters
        ----------
        n: int | None = None
            How many samples to take, None means jsut a single one, not in a list

        Returns
        -------
        C
            Get back a possible Config to use
        """
        ...

    @property
    @abstractmethod
    def space(self) -> ConfigurationSpace:
        """
        Returns
        -------
        ConfigurationSpace
        """
        ...
