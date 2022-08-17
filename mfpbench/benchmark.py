from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, overload

from ConfigSpace import ConfigurationSpace
from numpy.random import RandomState

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

    @abstractmethod
    def query(self, config: C, fidelity: F) -> R:
        """Submit a query and get a result

        Parameters
        ----------
        query: C
            The query to use

        fidelity: F
            The fidelity at which to query

        Returns
        -------
        R
            The result of the query
        """
        ...

    @abstractmethod
    def trajectory(self, config: C) -> list[R]:
        """Get the full trajectory of a configuration

        Parameters
        ----------
        config : C
            The config to query

        Returns
        -------
        list[R]
            A list of the results
        """
        ...

    # No number specified, just return one config
    @overload
    def sample(self, n: None = None, *, seed: int | RandomState | None = ...) -> C:
        ...

    # With a number, return many in a list
    @overload
    def sample(self, n: int, *, seed: int | RandomState | None = ...) -> list[C]:
        ...

    @abstractmethod
    def sample(
        self,
        n: int | None = None,
        *,
        seed: int | RandomState | None = None,
    ) -> C | list[C]:
        """Sample a random possible config

        Parameters
        ----------
        seed : int | RandomState | None = None
            The seed to use

        n: int | None = None
            How many samples to take, None means jsut a single one, not in a list

        Returns
        -------
        C
            Get back a possible Config to use
        """
        ...

    @abstractmethod
    def configspace(self, seed: int | RandomState | None = None) -> ConfigurationSpace:
        """
        Returns
        -------
        ConfigurationSpace
            Get the ConfigurationSpace for this benchmark
        """
        ...
