from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Iterator, Mapping, TypeVar, overload

import numpy as np

from mfpbench.config import Config
from mfpbench.result import Result
from mfpbench.resultframe import ResultFrame

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

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

    fidelity_range: tuple[F, F, F]
    """The fidelity range of this benchmark, (start, end, step)"""

    start: F
    """The start of the fidelity range"""

    end: F
    """The end of the fidelity range"""

    step: F
    """The step of the fidelity range"""

    fidelity_name: str
    """The name of the fidelity used in this benchmark"""

    space: ConfigurationSpace
    """The configuration space used in this benchmark"""

    Config: type[C]
    """The config type of this benchmark"""

    Result: type[R]
    """The result type of this benchmark"""

    has_conditionals: bool = False
    """Whether this benchmark has conditionals in it or not"""

    _default_prior_dir = PRIOR_DIR
    """The default directory for priors"""

    def __init__(
        self,
        name: str,
        space: ConfigurationSpace,
        *,
        seed: int | None = None,
        prior: str | Path | C | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
    ):
        """Initialize the benchmark.

        Args:
            name: The name of this benchmark
            space: The configuration space to use for the benchmark.
            seed: The seed to use.
            prior: The prior to use for the benchmark. If None, no prior is used.
                If a str, will check the local location first for a prior
                specific for this benchmark, otherwise assumes it to be a Path.
                If a Path, will load the prior from the path.
                If a Mapping, will be used directly.
            perturb_prior: If not None, will perturb the prior by this amount.
                For numericals, this is interpreted as the standard deviation of a
                normal distribution while for categoricals, this is interpreted
                as the probability of swapping the value for a random one.
        """
        self.name = name
        self.seed = seed
        self.space = space
        self.start: F = self.fidelity_range[0]
        self.end: F = self.fidelity_range[1]
        self.step: F = self.fidelity_range[2]

        self._prior_arg = prior

        # NOTE: This is handled entirely by subclasses as it requires knowledge
        # of the overall space the prior comes from, which only the subclasses now
        # at construction time. There's probably a better way to handle this but
        # for now this is fine.
        if perturb_prior is not None and not (0 <= perturb_prior <= 1):
            raise NotImplementedError(
                "If perturbing prior, `perturb_prior` must be in [0, 1]",
            )

        self.perturb_prior = perturb_prior
        self.prior: C | None = None

        if prior is not None:
            self.prior = self._load_prior(prior, benchname=self.name)
            self.prior.validate()
        else:
            self.prior = None

        if self.prior is not None and self.perturb_prior is not None:
            self.prior = self.prior.perturb(
                space,
                seed=self.seed,
                std=self.perturb_prior,
                categorical_swap_chance=self.perturb_prior,
            )

        if self.prior is not None:
            self.prior.set_as_default_prior(space)

    @classmethod
    def _load_prior(
        cls,
        prior: str | Path | Mapping[str, Any] | C,
        benchname: str | None = None,
    ) -> C:
        Config: type[C] = cls.Config  # Need to be a bit explicit here

        if isinstance(prior, str):
            # It's a str, use as a key into available priors
            if benchname is not None:
                assumed_path = cls._default_prior_dir / f"{benchname}-{prior}.yaml"
                if assumed_path.exists():
                    return Config.from_file(assumed_path)

            # Else we consider the prior to be a str reprsenting a Path
            return Config.from_file(Path(prior))

        if isinstance(prior, Path):
            return Config.from_file(prior)

        if isinstance(prior, Config):
            return prior

        if isinstance(prior, Mapping):
            return Config.from_dict(prior)

        raise ValueError(f"Unknown prior type {type(prior)}")

    def iter_fidelities(
        self,
        frm: F | None = None,
        to: F | None = None,
        step: F | None = None,
    ) -> Iterator[F]:
        """Iterate through the advertised fidelity space of the benchmark.

        Args:
            frm: Start of the curve, defaults to the minimum fidelity
            to: End of the curve, defaults to the maximum fidelity
            step: Step size, defaults to benchmark standard (1 for epoch)

        Returns:
            An iterator over the fidelities
        """
        frm = frm if frm is not None else self.start
        to = to if to is not None else self.end
        step = step if step is not None else self.step
        assert self.start <= frm <= to <= self.end

        dtype = int if isinstance(frm, int) else float
        fidelities: list[F] = list(
            np.arange(start=frm, stop=(to + step), step=step, dtype=dtype),
        )

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

    def query(
        self,
        config: C | Mapping[str, Any],
        at: F | None = None,
        *,
        argmax: str | None = None,
        argmin: str | None = None,
    ) -> R:
        """Submit a query and get a result.

        Args:
            config: The query to use
            at: The fidelity at which to query, defaults to None which means *maximum*
            argmax: Whether to return the argmax up to the point `at`. Will be slower as
                it has to get the entire trajectory. Uses the key from the Results.
            argmin: Whether to return the argmin up to the point `at`. Will be slower as
                it has to get the entire trajectory. Uses the key from the Results.

        Returns:
            The result of the query
        """
        at = at if at is not None else self.end
        assert self.start <= at <= self.end

        if argmax is not None and argmin is not None:
            raise ValueError("Can't have both argmax and argmin")

        if argmax is not None:
            _argmax = argmax
            return max(
                self.trajectory(config, frm=self.start, to=at),
                key=lambda r: getattr(r, _argmax),
            )

        if argmin is not None:
            _argmin = argmin
            return min(
                self.trajectory(config, frm=self.start, to=at),
                key=lambda r: getattr(r, _argmin),
            )

        if not isinstance(config, self.Config):
            _config = self.Config.from_dict(config)
        else:
            _config = config

        return self._objective_function(_config, at=at)

    def trajectory(
        self,
        config: C | Mapping[str, Any],
        *,
        frm: F | None = None,
        to: F | None = None,
        step: F | None = None,
    ) -> list[R]:
        """Get the full trajectory of a configuration.

        Args:
            config: The config to query
            frm: Start of the curve, should default to the start
            to: End of the curve, should default to the total
            step: Step size, defaults to ``cls.default_step``

        Returns:
            A list of the results for this config
        """
        to = to if to is not None else self.end
        frm = frm if frm is not None else self.start
        step = step if step is not None else self.step

        if not isinstance(config, self.Config):
            _config = self.Config.from_dict(config)
        else:
            _config = config

        return self._trajectory(_config, frm=frm, to=to, step=step)

    @abstractmethod
    def _objective_function(self, config: C, *, at: F) -> R:
        """Get the value of the benchmark for a config at a fidelity.

        Args:
            config: The config to query
            at: The fidelity to get the result at

        Returns:
            The result of the config
        """
        ...

    def _trajectory(self, config: C, *, frm: F, to: F, step: F) -> list[R]:
        """Get the trajectory of a config.

        By default this will just call the
        [`_objective_function()`][mfpbench.Benchmark._objective_function] for
        each fidelity but this can be overwritten if this can be done more optimaly.

        Args:
            config: The config to query
            frm: Start of the curve.
            to: End of the curve.
            step: Step size.

        Returns:
            A list of the results for this config
        """
        return [
            self._objective_function(config, at=fidelity)
            for fidelity in self.iter_fidelities(frm=frm, to=to, step=step)
        ]

    # No number specified, just return one config
    @overload
    def sample(
        self,
        n: None = None,
        *,
        seed: int | np.random.RandomState | None = None,
    ) -> C:
        ...

    # With a number, return many in a list
    @overload
    def sample(
        self,
        n: int,
        *,
        seed: int | np.random.RandomState | None = None,
    ) -> list[C]:
        ...

    def sample(
        self,
        n: int | None = None,
        *,
        seed: int | np.random.RandomState | None = None,
    ) -> C | list[C]:
        """Sample a random possible config.

        Args:
            n: How many samples to take, None means jsut a single one, not in a list
            seed: The seed to use for sampling

                !!! note "Seeding"

                    This is different than any seed passed to the construction
                    of the benchmark.

        Returns:
            Get back a possible Config to use
        """
        space = copy.deepcopy(self.space)
        if isinstance(seed, np.random.RandomState):
            rng = seed.randint(0, 2**32 - 1)
        else:
            rng = (
                seed
                if seed is not None
                else np.random.default_rng().integers(0, 2**32 - 1)
            )

        space.seed(rng)
        if n is None:
            return self.Config.from_dict(space.sample_configuration())

        # Just because of how configspace works
        if n == 1:
            return [self.Config.from_dict(space.sample_configuration())]

        return [self.Config.from_dict(c) for c in space.sample_configuration(n)]

    def frame(self) -> ResultFrame[C, F, R]:
        """Get an empty frame to record with."""
        return ResultFrame[C, F, R]()
