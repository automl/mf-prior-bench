from __future__ import annotations

from abc import ABC
from typing import Any

from pathlib import Path

import jahs_bench
from ConfigSpace import Configuration, ConfigurationSpace

from mfpbench.benchmark import Benchmark
from mfpbench.download import DATAROOT
from mfpbench.jahs.config import JAHSConfig
from mfpbench.jahs.result import JAHSResult
from mfpbench.jahs.spaces import jahs_configspace
from mfpbench.util import rename


class JAHSBenchmark(Benchmark, ABC):
    """Manages access to jahs-bench

    Use one of the concrete classes below to access a specific version:
    * JAHSCifar10
    * JAHSColorectalHistology:
    * JAHSFashionMNIST:
    """

    Config = JAHSConfig
    Result = JAHSResult
    fidelity_name = "epoch"
    fidelity_range = (1, 200, 1)

    task: jahs_bench.BenchmarkTasks

    # Where the data for jahsbench should be located relative to the path given
    _default_download_dir: Path = DATAROOT / "jahs-bench-data"
    _result_renames = {
        "size_MB": "size",
        "FLOPS": "flops",
        "valid-acc": "valid_acc",
        "test-acc": "test_acc",
        "train-acc": "train_acc",
    }

    def __init__(
        self,
        *,
        datadir: str | Path | None = None,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        datadir : str | Path | None = None
            The path to where mfpbench stores it data. If left to default (None), will
            use the `_default_download_dir = ./data/jahs-bench-data`.

        seed : int | None = None
            The seed to give this benchmark instance
        """
        super().__init__(seed=seed)

        if datadir is None:
            datadir = JAHSBenchmark._default_download_dir

        self.datadir = Path(datadir) if isinstance(datadir, str) else datadir
        if not self.datadir.exists():
            raise FileNotFoundError(
                f"Can't find folder at {self.datadir}, have you run\n"
                f"`python -m mfpbench.download --data-dir {self.datadir.parent}`"
            )

        # Loaded on demand with `@property`
        self._bench: jahs_bench.Benchmark | None = None
        self._configspace = jahs_configspace(self.seed)

    # explicit overwrite
    def load(self) -> None:
        """Pre-load JAHS XGBoost model before querying the first time"""
        self._bench = jahs_bench.Benchmark(
            task=self.task,
            save_dir=self.datadir,
            download=False,
        )

    @property
    def bench(self) -> jahs_bench.Benchmark:
        """The underlying benchmark used"""
        if not self._bench:
            self._bench = jahs_bench.Benchmark(
                task=self.task,
                save_dir=self.datadir,
                download=False,
            )

        return self._bench

    @property
    def space(self) -> ConfigurationSpace:
        """The ConfigurationSpace for this benchmark"""
        return self._configspace

    def query(
        self,
        config: JAHSConfig | dict[str, Any] | Configuration,
        at: int | None = None,
    ) -> JAHSResult:
        """Query the results for a config

        Parameters
        ----------
        config : JAHSConfig | dict[str, Any] | Configuration
            The config to query

        at : int | None = None
            The epoch at which to query at, defaults to max (200) if left as None

        Returns
        -------
        JAHSResult
            The result for the config at the given epoch
        """
        at = at if at is not None else self.end
        assert self.start <= at <= self.end

        if isinstance(config, Configuration):
            config = {**config}

        if isinstance(config, JAHSConfig):
            config = config.dict()

        results = self.bench.__call__(config, nepochs=at)
        result = results[at]

        return self.Result(
            config=self.Config(**config),  # Just make sure it's a JAHSConfig
            fidelity=at,
            **rename(result, keys=self._result_renames),
        )

    def trajectory(
        self,
        config: JAHSConfig | dict[str, Any] | Configuration,
        *,
        frm: int | None = None,
        to: int | None = None,
        step: int | None = None,
    ) -> list[JAHSResult]:
        """Query the trajectory of a config as it ranges over a fidelity

        Parameters
        ----------
        config : JAHSConfig | dict[str, Any] | Configuration
            The config to query

        frm: int | None = None
            Start of the curve, defaults to the minimum fidelity (1)

        to: int | None = None
            End of the curve, defaults to the maximum fidelity (200)

        step: int | None = None
            Step size, defaults to benchmark standard (1 for epoch)

        Returns
        -------
        list[JAHSResult]
            The results over that trajectory
        """
        to = to if to is not None else self.end

        if isinstance(config, Configuration):
            config = {**config}

        if isinstance(config, JAHSConfig):
            config = config.dict()

        try:
            results = self.bench.__call__(config, nepochs=to, full_trajectory=True)
        except TypeError:
            # See: https://github.com/automl/jahs_bench_201/issues/5
            results = {
                f: self.bench.__call__(config, nepochs=f)[f]
                for f in self.iter_fidelities(frm=frm, to=to, step=step)
            }

        return [
            self.Result(
                config=self.Config(**config),
                fidelity=i,
                **rename(results[i], keys=self._result_renames),
            )
            for i in self.iter_fidelities(frm=frm, to=to, step=step)
        ]


class JAHSCifar10(JAHSBenchmark):
    task = jahs_bench.BenchmarkTasks.CIFAR10


class JAHSColorectalHistology(JAHSBenchmark):
    task = jahs_bench.BenchmarkTasks.ColorectalHistology


class JAHSFashionMNIST(JAHSBenchmark):
    task = jahs_bench.BenchmarkTasks.FashionMNIST
