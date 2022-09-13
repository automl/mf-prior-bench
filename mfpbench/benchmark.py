from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Iterator, TypeVar, overload

from pathlib import Path

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from mfpbench.config import Config
from mfpbench.result import Result
from mfpbench.resultframe import ResultFrame

# The kind of Config to the benchmark
C = TypeVar("C", bound=Config)

# The return value from a config query
R = TypeVar("R", bound=Result)

# The kind of fidelity used in the benchmark
F = TypeVar("F", int, float)


class Benchmark(Generic[C, R, F], ABC):
    """Base class for a Benchmark"""

    # The fidelity range and name of this benchmark
    fidelity_range: tuple[F, F, F]
    fidelity_name: str

    # The config and result type of this benchmark
    Config: type[C]
    Result: type[R]

    # The priors available for this benchmark
    available_priors: dict[str, C] | None = None
    _default_prior: C | None = None

    # Whether this benchmark has conditonals in it or not
    has_conditionals: bool = False

    def __init__(
        self,
        seed: int | None = None,
        prior: str | Path | C | dict[str, Any] | Configuration | None = None,
    ):
        self.seed = seed
        self.start: F = self.fidelity_range[0]
        self.end: F = self.fidelity_range[1]
        self.step: F = self.fidelity_range[2]

        self._prior_arg = prior

        self.prior: C | None
        if prior is not None:
            # It's a str, use as a key into available priors
            if isinstance(prior, str):
                if self.available_priors is None:
                    clsname = {self.__class__.__name__}
                    raise ValueError(f"{clsname} has no prior called {prior}.")

                retrieved = self.available_priors.get(prior)
                if retrieved is None:
                    raise KeyError(f"{prior} not in {self.available_priors}")

                self.prior = retrieved

            elif isinstance(prior, Path):
                self.prior = self.Config.from_file(prior)

            elif isinstance(prior, dict):
                self.prior = self.Config.from_dict(prior)

            elif isinstance(prior, Configuration):
                self.prior = self.Config.from_configuration(**prior)

            else:
                self.prior = prior

        # no prior, use default
        else:
            if self.available_priors is not None:
                assert self._default_prior is not None, "No default prior?"

            self.prior = self._default_prior

        # Whatever prior we end up with, make sure it's valid
        if self.prior is not None:
            self.prior.validate()

    def iter_fidelities(
        self,
        frm: F | None = None,
        to: F | None = None,
        step: F | None = None,
    ) -> Iterator[F]:
        """Iterate through the advertised fidelity space of the benchmark

        Parameters
        ----------
        frm: F | None = None
            Start of the curve, defaults to the minimum fidelity

        to: F | None = None
            End of the curve, defaults to the maximum fidelity

        step: F | None = None
            Step size, defaults to benchmark standard (1 for epoch)

        Returns
        -------
        Iterator[F]
            Returns an iterator over the fidelities
        """
        frm = frm if frm is not None else self.start
        to = to if to is not None else self.end
        step = step if step is not None else self.step
        assert self.start <= frm <= to <= self.end

        dtype = int if isinstance(frm, int) else float
        fidelities = np.arange(start=frm, stop=(to + step), step=step, dtype=dtype)

        # Note: Clamping floats on arange
        #
        #   There's an annoying detail about floats here, essentially we could over
        #   (frm=0.03, to + step = 1+ .05, step=0.5) -> [0.03, 0.08, ..., 1.03]
        #   We just clamp this to the last fidelity
        #
        #   This does not effect ints
        if isinstance(step, float) and fidelities[-1] >= self.end:
            fidelities[-1] = self.end

        yield from fidelities

    def load(self) -> None:
        """Explicitly load the benchmark before querying, optional"""
        pass

    @abstractmethod
    def query(self, config: C | dict | Configuration, at: F | None = None) -> R:
        """Submit a query and get a result

        Parameters
        ----------
        config: C | dict | Configuration
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
        if n is None:
            return self.Config.from_dict(self.space.sample_configuration())

        # Just because of how configspace works
        elif n == 1:
            return [self.Config.from_dict(self.space.sample_configuration())]

        else:
            return [
                self.Config.from_dict(c) for c in self.space.sample_configuration(n)
            ]

    @property
    @abstractmethod
    def space(self) -> ConfigurationSpace:
        """The configuration space for this benchmark, incorporating the prior if given

        Returns
        -------
        ConfigurationSpace
        """
        ...

    def frame(self) -> ResultFrame[C, F, R]:
        """Get an empty frame to record with"""
        return ResultFrame[C, F, R]()
