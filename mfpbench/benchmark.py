from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Iterator, TypeVar, overload

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from mfpbench.config import Config
from mfpbench.result import Result
from mfpbench.resultframe import ResultFrame

HERE = Path(__file__).parent.parent
PRIOR_DIR = HERE / "priors"

# The kind of Config to the benchmark
C = TypeVar("C", bound=Config)

# The return value from a config query
R = TypeVar("R", bound=Result)

# The kind of fidelity used in the benchmark
F = TypeVar("F", int, float)


class Benchmark(Generic[C, R, F], ABC):
    """Base class for a Benchmark."""

    # The fidelity range and name of this benchmark
    fidelity_range: tuple[F, F, F]
    fidelity_name: str

    # The config and result type of this benchmark
    Config: type[C]
    Result: type[R]

    # Whether this benchmark has conditonals in it or not
    has_conditionals: bool = False

    # Where the repo's preset priors are located
    _default_prior_dir = PRIOR_DIR

    def __init__(
        self,
        seed: int | None = None,
        prior: str | Path | C | dict[str, Any] | Configuration | None = None,
        perturb_prior: float | None = None,
        **kwargs: Any,  # pyright: ignore
    ):
        self.seed = seed
        self.start: F = self.fidelity_range[0]
        self.end: F = self.fidelity_range[1]
        self.step: F = self.fidelity_range[2]

        self._prior_arg = prior

        # NOTE: This is handled entirely by subclasses as it requires knowledge
        # of the overall space the prior comes from, which only the subclasses now
        # at construction time. There's probably a better way to handle this but
        # for now this is fine.
        if perturb_prior is not None and not (0 <= perturb_prior < 1):
            raise NotImplementedError(
                "If perturbing prior, `perturb_prior` must be in (0, 1]"
            )
        self.perturb_prior = perturb_prior

        self.prior: C | None
        if prior is not None:
            # It's a str, use as a key into available priors
            if isinstance(prior, str):
                assumed_path = self._default_prior_dir / f"{self.basename}-{prior}.yaml"
                if assumed_path.exists():
                    self.prior = self.Config.from_file(assumed_path)
                else:
                    # Else we consider the prior to be a str reprsenting a Path
                    self.prior = self.Config.from_file(Path(prior))

            elif isinstance(prior, Path):
                self.prior = self.Config.from_file(prior)

            elif isinstance(prior, dict):
                self.prior = self.Config.from_dict(prior)

            elif isinstance(prior, Configuration):
                self.prior = self.Config.from_configuration(prior)

            else:
                self.prior = prior

            self.prior.validate()

        else:
            self.prior = None

    @property
    @abstractmethod
    def basename(self) -> str:
        """A basename used for identifying pregenerated prior files."""
        ...

    def iter_fidelities(
        self,
        frm: F | None = None,
        to: F | None = None,
        step: F | None = None,
    ) -> Iterator[F]:
        """Iterate through the advertised fidelity space of the benchmark.

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
        """Explicitly load the benchmark before querying, optional."""
        pass

    @abstractmethod
    def query(
        self,
        config: C | dict | Configuration,
        at: F | None = None,
        *,
        argmax: bool = False,
    ) -> R:
        """Submit a query and get a result.

        Parameters
        ----------
        config: C | dict | Configuration
            The query to use

        at: F | None = None
            The fidelity at which to query, defaults to None which means *maximum*

        argmax: bool = False
            Whether to return the argmax up to the point `at`. Will be slower as it
            has to get the entire trajectory. Uses the corresponding Result's score.

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
        """Get the full trajectory of a configuration.

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
        """Sample a random possible config.

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
        """The configuration space for this benchmark, incorporating the prior if given.

        Returns
        -------
        ConfigurationSpace
        """
        ...

    def frame(self) -> ResultFrame[C, F, R]:
        """Get an empty frame to record with."""
        return ResultFrame[C, F, R]()
