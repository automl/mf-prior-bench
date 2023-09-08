"""The hartmann benchmarks.

The presets of terrible, bad, moderate and good are empirically obtained hyperparameters
for the hartmann function

The function flattens with increasing fidelity bias.
Along with increasing noise, that obviously makes one config harder to distinguish from
another.
Moreover, this works with any number of fidelitiy levels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, Mapping, TypeVar
from typing_extensions import override

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from mfpbench.benchmark import Benchmark
from mfpbench.config import Config
from mfpbench.result import Result
from mfpbench.synthetic.hartmann.generators import (
    MFHartmann3,
    MFHartmann6,
    MFHartmannGenerator,
)


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class MFHartmann3Config(Config):
    X_0: float
    X_1: float
    X_2: float

    def validate(self) -> None:
        """Validate this config."""
        assert 0.0 <= self.X_0 <= 1.0
        assert 0.0 <= self.X_1 <= 1.0
        assert 0.0 <= self.X_2 <= 1.0


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class MFHartmann6Config(Config):
    X_0: float
    X_1: float
    X_2: float
    X_3: float
    X_4: float
    X_5: float

    def validate(self) -> None:
        """Validate this config."""
        assert 0.0 <= self.X_0 <= 1.0
        assert 0.0 <= self.X_1 <= 1.0
        assert 0.0 <= self.X_2 <= 1.0
        assert 0.0 <= self.X_3 <= 1.0
        assert 0.0 <= self.X_4 <= 1.0
        assert 0.0 <= self.X_5 <= 1.0


C = TypeVar("C", MFHartmann3Config, MFHartmann6Config)


@dataclass(frozen=True)  # type: ignore[misc]
class MFHartmannResult(Result[C, int]):
    value: float
    fid_cost: float

    @property
    def score(self) -> float:
        """The score of interest."""
        # TODO: what should be an appropriate score since flipping signs may not be
        #  adequate or meaningful. When is the property score used?
        # Hartmann functions have multiple minimas with the global valued at < 0
        # The function evaluates to a y-value that needs to be minimized
        #  https://www.sfu.ca/~ssurjano/hart3.html
        raise NotImplementedError("There's no meaninfgul score for Hartmann functions")

    @property
    def error(self) -> float:
        """The score of interest."""
        # TODO: verify
        # Hartmann functions have multiple minimas with the global valued at < 0
        # The function evaluates to a y-value that needs to be minimized
        #  https://www.sfu.ca/~ssurjano/hart3.html
        return self.value

    @property
    def test_score(self) -> float:
        """Just returns the score."""
        raise NotImplementedError("There's no meaninfgul score for Hartmann functions")

    @property
    def test_error(self) -> float:
        """Just returns the error."""
        return self.error

    @property
    def val_score(self) -> float:
        """Just returns the score."""
        raise NotImplementedError("There's no meaninfgul score for Hartmann functions")

    @property
    def val_error(self) -> float:
        """Just returns the error."""
        return self.error

    @property
    def cost(self) -> float:
        """Just retuns the fidelity."""
        # return self.fidelity
        return self.fid_cost


G = TypeVar("G", bound=MFHartmannGenerator)


class MFHartmannBenchmark(Benchmark, Generic[G, C]):
    mfh_dims: ClassVar[int]
    """How many dimensions there are to the Hartmann function."""

    mfh_suffix: ClassVar[str]
    """Suffix for the benchmark name"""

    Config: type[C]
    """The Config type for this mfhartmann benchmark."""

    Generator: type[G]
    """The underlying mfhartmann function generator."""

    mfh_bias_noise: ClassVar[tuple[float, float]] = (0.5, 0.1)
    """The default bias and noise for mfhartmann benchmarks."""

    fidelity_name = "z"
    fidelity_range = (3, 100, 1)
    Result = MFHartmannResult

    def __init__(
        self,
        *,
        seed: int | None = None,
        bias: float | None = None,
        noise: float | None = None,
        prior: str | Path | C | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
    ):
        """Initialize the benchmark.

        Args:
            seed: The seed to use.
            bias: How much bias to introduce
            noise: How much noise to introduce
            prior: The prior to use for the benchmark.

                * if `Path` - path to a file
                * if `Mapping` - Use directly
                * if `None` - There is no prior

            perturb_prior: If not None, will perturb the prior by this amount.
                For numericals, while for categoricals, this is interpreted as
                the probability of swapping the value for a random one.
        """
        cls = self.__class__
        self.bias = bias if bias is not None else cls.mfh_bias_noise[0]
        self.noise = noise if noise is not None else cls.mfh_bias_noise[1]
        self.mfh = cls.Generator(
            n_fidelities=cls.fidelity_range[1],
            fidelity_noise=self.noise,
            fidelity_bias=self.bias,
            seed=seed,
        )

        name = (
            f"mfh{cls.mfh_dims}_{cls.mfh_suffix}"
            if cls.mfh_suffix != ""
            else f"mfh{cls.mfh_dims}"
        )
        space = ConfigurationSpace(name=name, seed=seed)
        space.add_hyperparameters(
            [
                UniformFloatHyperparameter(f"X_{i}", lower=0.0, upper=1.0)
                for i in range(cls.mfh_dims)
            ],
        )
        super().__init__(
            name=name,
            space=space,
            seed=seed,
            prior=prior,
            perturb_prior=perturb_prior,
        )

    @override
    def _objective_function(self, config: C, *, at: int) -> MFHartmannResult[C]:
        query = config.dict()

        # It's important here that we still have X_0, X_1, ..., X_n
        # We strip out the numerical part and sort by that
        Xs = tuple(query[s] for s in sorted(query, key=lambda k: int(k.split("_")[-1])))
        value = self.mfh(z=at, Xs=Xs)
        cost = self._fidelity_cost(at)

        return self.Result.from_dict(
            config=config,
            fidelity=at,
            result={"value": value, "fid_cost": cost},
        )

    def _fidelity_cost(self, at: int) -> float:
        # λ(z) on Pg 18 from https://arxiv.org/pdf/1703.06240.pdf
        return 0.05 + (1 - 0.05) * (at / self.fidelity_range[1]) ** 2

    @property
    def optimum(self) -> C:
        """The optimum of the benchmark."""
        optimum = {f"X_{i}": x for i, x in enumerate(self.Generator.optimum)}
        return self.Config.from_dict(optimum)


# -----------
# MFHartmann3
# -----------
class MFHartmann3Benchmark(MFHartmannBenchmark):
    Generator = MFHartmann3
    Config = MFHartmann3Config
    mfh_dims = MFHartmann3.dims
    mfh_suffix = ""


class MFHartmann3BenchmarkTerrible(MFHartmann3Benchmark):
    mfh_bias_noise = (4.0, 5.0)
    mfh_suffix = "terrible"


class MFHartmann3BenchmarkBad(MFHartmann3Benchmark):
    mfh_bias_noise = (3.5, 4.0)
    mfh_suffix = "bad"


class MFHartmann3BenchmarkModerate(MFHartmann3Benchmark):
    mfh_bias_noise = (3.0, 3.0)
    mfh_suffix = "moderate"


class MFHartmann3BenchmarkGood(MFHartmann3Benchmark):
    mfh_bias_noise = (2.5, 2.0)
    mfh_suffix = "good"


# -----------
# MFHartmann6
# -----------
class MFHartmann6Benchmark(MFHartmannBenchmark):
    Generator = MFHartmann6
    Config = MFHartmann6Config
    mfh_dims = MFHartmann6.dims
    mfh_suffix = ""


class MFHartmann6BenchmarkTerrible(MFHartmann6Benchmark):
    mfh_bias_noise = (4.0, 5.0)
    mfh_suffix = "terrible"


class MFHartmann6BenchmarkBad(MFHartmann6Benchmark):
    mfh_bias_noise = (3.5, 4.0)
    mfh_suffix = "bad"


class MFHartmann6BenchmarkModerate(MFHartmann6Benchmark):
    mfh_bias_noise = (3.0, 3.0)
    mfh_suffix = "moderate"


class MFHartmann6BenchmarkGood(MFHartmann6Benchmark):
    mfh_bias_noise = (2.5, 2.0)
    mfh_suffix = "good"
