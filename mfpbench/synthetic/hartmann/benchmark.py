"""The hartmann benchmarks.

The presets of terrible, bad, moderate and good are empirically obtained hyperparameters
for the hartmann function
    ! have not tested ranking correlation yet
The function flattens with increasing fidelity bias.
Along with increasing noise, that obviously makes one config harder to distinguish from
another.
Moreover, this works with any number of fidelitiy levels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter

from mfpbench.benchmark import Benchmark
from mfpbench.synthetic.hartmann.config import (
    MFHartmann3Config,
    MFHartmann6Config,
    MFHartmannConfig,
)
from mfpbench.synthetic.hartmann.generators import (
    MFHartmann3,
    MFHartmann6,
    MFHartmannGenerator,
)
from mfpbench.synthetic.hartmann.result import MFHartmannResult

G = TypeVar("G", bound=MFHartmannGenerator)
C = TypeVar("C", bound=MFHartmannConfig)


class MFHartmannBenchmark(Benchmark, Generic[G, C]):

    # fidelity_range = (1, 5, 1)
    fidelity_range = (3, 100, 1)
    fidelity_name = "z"

    Config: type[C]
    Result = MFHartmannResult

    Generator: type[G]
    bias_noise: tuple[float, float] = (0.5, 0.1)

    # How many dimensions there are to the Hartmann function
    dims: int
    suffix: str

    def __init__(
        self,
        seed: int | None = None,
        bias: float | None = None,
        noise: float | None = None,
        prior: str | Path | C | dict[str, Any] | Configuration | None = None,
        noisy_prior: bool = False,
        prior_noise_scale: float = 0.125,
        perturb_prior: float | None = None,
    ):
        """Initialize the benchmark.

        Parameters
        ----------
        seed : int | None = None
            The seed to use.

        bias : float | None = None
            How much bias to introduce

        noise : float | None = None
            How much noise to introduce

        prior: str | Path | MFHartmannConfig | dict | Configuration | None = None
            The prior to use for the benchmark.
            * if str - A preset
            * if Path - path to a file
            * if dict, Config, Configuration - A config
            * None - Use the default if available

        noisy_prior: bool = False
            Whether to add noise to the prior

        prior_noise_scale: float = 0.125
            The scaling factor for noise added to the prior
            `noise = prior_noise_scale * np.random.random(size=...)`

        perturb_prior: float | None = None
            A synonm for the two arguments `noisy_prior` and `prior_noise_scale`.
            This takes precedence over prior_noise_scale
        """
        super().__init__(seed=seed, prior=prior, perturb_prior=perturb_prior)
        if self.prior is None and noisy_prior:
            raise ValueError("`noisy_prior = True` specified but no `prior` given")

        self.bias = bias if bias is not None else self.bias_noise[0]
        self.noise = noise if noise is not None else self.bias_noise[1]

        if noisy_prior or perturb_prior is not None:
            self.noisy_prior = True
            self.prior_noise_scale = (
                perturb_prior if perturb_prior is not None else prior_noise_scale
            )
        else:
            self.noisy_prior = False
            self.prior_noise_scale = prior_noise_scale

        self.mfh = self.Generator(
            n_fidelities=self.end,
            fidelity_noise=self.noise,
            fidelity_bias=self.bias,
            seed=self.seed,
        )
        # Create the configspace
        self._configspace = ConfigurationSpace(name=str(self), seed=self.seed)
        self._configspace.add_hyperparameters(
            [
                UniformFloatHyperparameter(f"X_{i}", lower=0.0, upper=1.0)
                for i in range(self.dims)
            ]
        )

        # Set the prior on the config space
        if self.prior is not None:

            # If some noise seed was passed, we add some noise to the prior
            if self.noisy_prior:
                # Create noise to add
                rng = np.random.default_rng(seed=self.seed)
                uniform = rng.uniform(low=-1, high=1, size=self.dims)
                noises = self.prior_noise_scale * uniform
                d = self.prior.dict()

                # We iterate through the prior and add noise, clipping incase
                new_prior = self.Config.from_dict(
                    {
                        k: float(np.clip(v + n, a_min=0, a_max=1))
                        for (k, v), n in zip(d.items(), noises)
                    }
                )
                self.prior = new_prior

            self.prior.set_as_default_prior(self._configspace)

    @property
    def basename(self) -> str:
        if self.suffix != "":
            return f"mfh{self.dims}_{self.suffix}"
        else:
            return f"mfh{self.dims}"

    def query(
        self,
        config: C | dict | Configuration,
        at: int | None = None,
        *,
        argmax: bool = False,
    ) -> MFHartmannResult[C]:
        """Submit a query and get a result.

        Parameters
        ----------
        config: C | dict | Configuration
            The query to use

        at: int | None = None
            The fidelity at which to query, defaults to None which means *maximum*

        argmax: bool = False
            Whether to return the argmax up to the point `at`. Will be slower as it
            has to get the entire trajectory. Uses the corresponding Result's score.

        Returns
        -------
        MFHartmannResult
            The result of the query
        """
        at = at if at is not None else self.end
        assert self.start <= at <= self.end

        if argmax:
            return max(self.trajectory(config, to=at), key=lambda r: r.score)

        if isinstance(config, (Configuration, dict)):
            config = self.Config.from_dict(config)

        assert isinstance(config, self.Config)

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

    def trajectory(
        self,
        config: C | dict | Configuration,
        *,
        frm: int | None = None,
        to: int | None = None,
        step: int | None = None,
    ) -> list[MFHartmannResult[C]]:
        """Get the full trajectory of a configuration.

        Parameters
        ----------
        config : C | dict | Configuration
            The config to query

        frm: int | None = None
            Start of the curve, should default to the start (1)

        to: int | None = None
            End of the curve, should default to the total (5)

        step: int | None = None
            Step size, defaults to 1

        Returns
        -------
        list[MFHartmannResult]
            A list of the results for this config
        """
        if isinstance(config, (Configuration, dict)):
            config = self.Config.from_dict(config)

        assert isinstance(config, self.Config)

        query = config.dict()

        # It's important here that we still have X_0, X_1, ..., X_n
        # We strip out the numerical part and sort by that
        Xs = tuple(query[s] for s in sorted(query, key=lambda k: int(k.split("_")[-1])))

        results_fidelities = [
            (self.mfh(z=f, Xs=Xs), f)
            for f in self.iter_fidelities(frm=frm, to=to, step=step)
        ]

        return [
            self.Result.from_dict(
                config=config,
                fidelity=f,
                result={"value": r, "fid_cost": self._fidelity_cost(f)},
            )
            for r, f in results_fidelities
        ]

    def _fidelity_cost(self, at: int) -> float:
        # λ(z) on Pg 18 from https://arxiv.org/pdf/1703.06240.pdf
        return 0.05 + (1 - 0.05) * (at / self.fidelity_range[1]) ** 2

    @property
    def space(self) -> ConfigurationSpace:
        return self._configspace

    @property
    def optimum(self) -> C:
        optimum = {f"X_{i}": x for i, x in enumerate(self.Generator.optimum)}
        return self.Config.from_dict(optimum)

    def __repr__(self) -> str:
        params = f"bias={self.bias}, noise={self.noise}, prior={self._prior_arg}"
        return f"{self.__class__.__name__}({params})"


# -----------
# MFHartmann3
# -----------
class MFHartmann3Benchmark(MFHartmannBenchmark):
    Generator = MFHartmann3
    Config = MFHartmann3Config
    dims = MFHartmann3.dims
    suffix = ""


class MFHartmann3BenchmarkTerrible(MFHartmann3Benchmark):
    bias_noise = (4.0, 5.0)
    suffix = "terrible"


class MFHartmann3BenchmarkBad(MFHartmann3Benchmark):
    bias_noise = (3.5, 4.0)
    suffix = "bad"


class MFHartmann3BenchmarkModerate(MFHartmann3Benchmark):
    bias_noise = (3.0, 3.0)
    suffix = "moderate"


class MFHartmann3BenchmarkGood(MFHartmann3Benchmark):
    bias_noise = (2.5, 2.0)
    suffix = "good"


# -----------
# MFHartmann6
# -----------
class MFHartmann6Benchmark(MFHartmannBenchmark):
    Generator = MFHartmann6
    Config = MFHartmann6Config
    dims = MFHartmann6.dims
    suffix = ""


class MFHartmann6BenchmarkTerrible(MFHartmann6Benchmark):
    bias_noise = (4.0, 5.0)
    suffix = "terrible"


class MFHartmann6BenchmarkBad(MFHartmann6Benchmark):
    bias_noise = (3.5, 4.0)
    suffix = "bad"


class MFHartmann6BenchmarkModerate(MFHartmann6Benchmark):
    bias_noise = (3.0, 3.0)
    suffix = "moderate"


class MFHartmann6BenchmarkGood(MFHartmann6Benchmark):
    bias_noise = (2.5, 2.0)
    suffix = "good"
