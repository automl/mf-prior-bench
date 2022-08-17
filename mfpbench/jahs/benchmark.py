from __future__ import annotations

from abc import ABC
from typing import overload

from pathlib import Path

import jahs_bench
from ConfigSpace import ConfigurationSpace
from numpy.random import RandomState

from mfpbench.benchmark import Benchmark
from mfpbench.download import DATAROOT
from mfpbench.jahs.config import JAHSConfig
from mfpbench.jahs.result import JAHSResult
from mfpbench.jahs.spaces import JAHSConfigspace


class JAHSBench(Benchmark[JAHSConfig, JAHSResult, int], ABC):
    """Manages access to jahs-bench

    Use one of the concrete classes below to access a specific version:
    * JAHSCifar10
    * JAHSColorectalHistology:
    * JAHSFashionMNIST:
    """

    task: jahs_bench.BenchmarkTasks
    name: str = "jahs-bench-data"
    max_epoch: int = 200

    def __init__(self, datadir: str | Path | None = None):
        if datadir is None:
            datadir = DATAROOT

        if isinstance(datadir, str):
            datadir = Path(datadir)

        save_dir = datadir / self.name

        self.bench = jahs_bench.Benchmark(
            task=self.task,
            save_dir=save_dir,
            download=True,
        )

    def query(
        self,
        config: JAHSConfig,
        fidelity: int = 200,
    ) -> JAHSResult:
        assert 1 <= fidelity <= 200

        results = self.bench.__call__(config.dict(), nepochs=fidelity)
        result = results[fidelity]

        return JAHSResult.from_dict(config, result, fidelity)

    def trajectory(self, config: JAHSConfig, *, to: int = 200) -> list[JAHSResult]:
        assert 1 <= to <= 200

        results = self.bench.__call__(config.dict(), nepochs=to, full_trajectory=True)

        return [JAHSResult.from_dict(config, results[i], i) for i in range(1, 201)]

    @overload
    def sample(self, n: None = None, *, seed: int | None | RandomState = ...) -> JAHSConfig:
        ...

    @overload
    def sample(self, n: int, *, seed: int | None | RandomState = ...) -> list[JAHSConfig]:
        ...

    def sample(
        self,
        n: int | None = None,
        *,
        seed: int | None | RandomState = None,
    ) -> JAHSConfig | list[JAHSConfig]:
        if n is None:
            config = self.configspace(seed=seed).sample_configuration()
            return JAHSConfig(**config)
        else:
            configs = self.configspace(seed=seed).sample_configuration(n)
            return [JAHSConfig(**config) for config in configs]

    def configspace(self, seed: int | RandomState | None = None) -> ConfigurationSpace:
        return JAHSConfigspace(seed=seed)


class JAHSCifar10(JAHSBench):
    task = jahs_bench.BenchmarkTasks.CIFAR10


class JAHSColorectalHistology(JAHSBench):
    task = jahs_bench.BenchmarkTasks.ColorectalHistology


class JAHSFashionMNIST(JAHSBench):
    task = jahs_bench.BenchmarkTasks.FashionMNIST
