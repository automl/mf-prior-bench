"""
The presets of terrible, bad, moderate and good are empirically obtained hyperparameters
for the hartmann function
    ! have not tested ranking correlation yet
The function flattens with increasing fidelity bias.
Along with increasing noise, that obviously makes one config harder to distinguish from
another.
Moreover, this works with any number of fidelitiy levels
"""
from __future__ import annotations

from typing import Generic, TypeVar

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
from mfpbench.synthetic.hartmann.hartmann_priors import HARTMANN3D, HARTMANN6D


G = TypeVar("G", bound=MFHartmannGenerator)
C = TypeVar("C", bound=MFHartmannConfig)


class MFHartmannBenchmark(Benchmark, Generic[G, C]):

    # fidelity_range = (1, 5, 1)
    fidelity_range = (3, 100, 1)  # scaling original fidelity space [0, 1] to [1, 10]
    fidelity_name = "z"

    Result = MFHartmannResult
    Config: type[C]

    Generator: type[G]
    bias_noise: tuple[float, float] = (0.5, 0.1)

    def __init__(
        self,
        seed: int | None = None,
        bias: float | None = None,
        noise: float | None = None,
        prior: str = None
    ):
        """
        Parameters
        ----------
        seed : int | None = None
            The seed to use

        bias : float | None

        noise : float | None

        """
        super().__init__(seed)
        self.bias = bias if bias is not None else self.bias_noise[0]
        self.noise = noise if noise is not None else self.bias_noise[1]
        self.prior = prior
        self.mfh = self.Generator(
            n_fidelities=self.end,
            fidelity_noise=self.noise,
            fidelity_bias=self.bias,
            seed=self.seed,
        )
        self._configspace: ConfigurationSpace | None = None

    def query(
        self,
        config: C | dict | Configuration,
        at: int | None = None,
    ) -> MFHartmannResult[C]:
        """Submit a query and get a result

        Parameters
        ----------
        config: C | dict | Configuration
            The query to use

        at: int | None = None
            The fidelity at which to query, defaults to None which means *maximum*

        Returns
        -------
        MFHartmannResult
            The result of the query
        """
        at = at if at is not None else self.end
        assert self.start <= at <= self.end

        if isinstance(config, Configuration):
            config = {**config}

        if isinstance(config, MFHartmannConfig):
            config = config.dict()

        assert isinstance(config, dict), "I assume this is the case by here?"

        # It's important here that we still have X_0, X_1, ..., X_n
        # We strip out the numerical part and sort by that
        Xs = tuple(
            config[s] for s in sorted(config, key=lambda k: int(k.split("_")[-1]))
        )
        log_z = np.log(at)
        log_lb, log_ub = np.log([self.fidelity_range[0], self.fidelity_range[1]])
        log_z_scaled = (log_z - log_lb) / (log_ub - log_lb)
        result = self.mfh(z=log_z_scaled, Xs=Xs)

        # return self.Result(
        #     config=self.Config(**config),
        #     fidelity=at,
        #     **{"value": result},
        # )

        # λ(z) on Pg 18 from https://arxiv.org/pdf/1703.06240.pdf
        cost = 0.05 + (1 - 0.05) * (at / self.fidelity_range[1]) ** 2
        return self.Result(
            config=self.Config(**config),
            fidelity=at,
            **{
                "value": result,
                "fid_cost": cost
            },
        )

    def trajectory(
        self,
        config: C | dict | Configuration,
        *,
        frm: int | None = None,
        to: int | None = None,
        step: int | None = None,
    ) -> list[MFHartmannResult[C]]:
        """Get the full trajectory of a configuration

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
        if isinstance(config, Configuration):
            config = {**config}

        if isinstance(config, MFHartmannConfig):
            config = config.dict()

        assert isinstance(config, dict), "I assume this is the case by here?"

        # It's important here that we still have X_0, X_1, ..., X_n
        # We strip out the numerical part and sort by that
        Xs = tuple(
            config[s] for s in sorted(config, key=lambda k: int(k.split("_")[-1]))
        )

        fidelities = list(self.iter_fidelities())
        results_fidelities = [(self.mfh(z=f, Xs=Xs), f) for f in fidelities]

        return [
            self.Result(config=self.Config(**config), fidelity=f, **{"value": r})
            for r, f in results_fidelities
        ]

    @property
    def space(self) -> ConfigurationSpace:
        """
        Returns
        -------
        ConfigurationSpace
        """
        if self.prior == "good":
            defaults = HARTMANN6D.GOOD_PRIOR if self.mfh.dims == 6 else HARTMANN3D.GOOD_PRIOR
        elif self.prior == "bad":
            defaults = HARTMANN6D.BAD_PRIOR if self.mfh.dims == 6 else HARTMANN3D.BAD_PRIOR
        else:
            defaults = HARTMANN6D.DEFAULT if self.mfh.dims == 6 else HARTMANN3D.DEFAULT

        if self._configspace is None:
            cs = ConfigurationSpace(
                name=f"{self.__class__.__name__}(bias={self.bias}, noise={self.noise})",
                seed=self.seed,
            )
            cs.add_hyperparameters(
                [
                    UniformFloatHyperparameter(
                        f"X_{i}", lower=0.0, upper=1.0, default_value=defaults[f"X_{i}"]
                    ) for i in range(self.mfh.dims)
                ]
            )
            self._configspace = cs

        return self._configspace


# -----------
# MFHartmann3
# -----------
class MFHartmann3Benchmark(MFHartmannBenchmark):
    Generator = MFHartmann3
    Config = MFHartmann3Config


class MFHartmann3BenchmarkTerrible(MFHartmann3Benchmark):
    bias_noise = (4.0, 0.8)


class MFHartmann3BenchmarkBad(MFHartmann3Benchmark):
    bias_noise = (2.0, 0.4)


class MFHartmann3BenchmarkModerate(MFHartmann3Benchmark):
    bias_noise = (1.0, 0.2)


class MFHartmann3BenchmarkGood(MFHartmann3Benchmark):
    bias_noise = (0.5, 0.1)


# -----------
# MFHartmann6
# -----------
class MFHartmann6Benchmark(MFHartmannBenchmark):
    Generator = MFHartmann6
    Config = MFHartmann6Config


class MFHartmann6BenchmarkTerrible(MFHartmann6Benchmark):
    bias_noise = (4.0, 0.8)


class MFHartmann6BenchmarkBad(MFHartmann6Benchmark):
    bias_noise = (2.0, 0.4)


class MFHartmann6BenchmarkModerate(MFHartmann6Benchmark):
    bias_noise = (1.0, 0.2)


class MFHartmann6BenchmarkGood(MFHartmann6Benchmark):
    bias_noise = (0.5, 0.1)
