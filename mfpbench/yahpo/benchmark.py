from __future__ import annotations

from typing import Any, TypeVar

from pathlib import Path

import yahpo_gym
from ConfigSpace import Configuration, ConfigurationSpace

from mfpbench.benchmark import Benchmark
from mfpbench.download import DATAROOT
from mfpbench.util import remove_hyperparameter
from mfpbench.yahpo.config import YAHPOConfig
from mfpbench.yahpo.result import YAHPOResult

_YAHPO_LOADED = False


def _ensure_yahpo_config_set(path: Path) -> None:
    if _YAHPO_LOADED:
        return

    yahpo_gym.local_config.init_config()
    yahpo_gym.local_config.set_data_path(str(path))
    return


# A Yahpo Benchmark is parametrized by a YAHPOConfig, YAHPOResult and fidelity
C = TypeVar("C", bound=YAHPOConfig)
R = TypeVar("R", bound=YAHPOResult)
F = TypeVar("F", int, float)


class YAHPOBenchmark(Benchmark[C, R, F]):

    name: str  # Name of the benchmark
    instances: list[str] | None  # List of instances if any
    fidelity_name: str  # Name of the fidelity used

    # Where the data for yahpo gym data should be located relative to the path given
    _default_download_dir: Path = DATAROOT / "yahpo-gym-data"

    def __init__(
        self,
        task_id: str | None = None,
        *,
        datadir: str | Path | None = None,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        task_id: str
            The task id to choose from, see cls.instances

        datadir : str | Path | None = None
            The path to where mfpbench stores it data. If left to default (None), will
            use the `_default_download_dir = ./data/yahpo-gym-data`.

        seed : int | None = None
            The seed for the benchmark instance
        """
        # Validation
        cls = self.__class__
        if task_id is None:
            if self.instances is not None:
                raise ValueError(f"{cls} requires a task in {self.instances}")
        else:
            if self.instances is None:
                raise ValueError(f"{cls} no instances, you passed {task_id}")
            elif task_id not in self.instances:
                raise ValueError(f"{cls} requires a task in {self.instances}")

        super().__init__(seed=seed)
        if datadir is None:
            datadir = self._default_download_dir

        if isinstance(datadir, str):
            datadir = Path(datadir)

        datadir = Path(datadir) if isinstance(datadir, str) else datadir
        if not datadir.exists():
            raise FileNotFoundError(
                f"Can't find folder at {datadir}, have you run\n"
                f"`python -m mfpbench.download --data-dir {datadir.parent}`"
            )
        _ensure_yahpo_config_set(datadir)

        bench = yahpo_gym.BenchmarkSet(self.name, instance=task_id)

        # These can have one or two fidelities
        space = bench.get_opt_space(drop_fidelity_params=True, seed=seed)
        space = remove_hyperparameter("OpenML_task_id", space)

        self.bench = bench
        self.datadir = datadir
        self.task_id = task_id
        self._configspace = space

    def query(
        self,
        config: C | dict[str, Any] | Configuration,
        at: F | None = None,
    ) -> R:
        """Query the results for a config

        Parameters
        ----------
        config : C | dict[str, Any] | Configuration
            The config to query

        at : F | None = None
            The fidelity at which to query, defaults to None which means *maximum*

        Returns
        -------
        R
            The result for the config at the given epoch
        """
        at = at if at is not None else self.end
        assert self.start <= at <= self.end

        if isinstance(config, Configuration):
            config = {**config}

        if isinstance(config, YAHPOConfig):
            config = config.dict()

        assert isinstance(config, dict), "I assume this is the case by here?"

        if self.fidelity_name in config:
            msg = f"``query(config, at=...)`` for fidelity, not the config {config}"
            raise ValueError(msg)

        query: dict = {**config, self.fidelity_name: at}

        if self.task_id is not None:
            query["OpenML_task_id"] = self.task_id

        results: list[dict] = self.bench.objective_function(query, seed=self.seed)
        result = results[0]

        return self.Result.from_dict(
            config=self.Config(**config), result=result, fidelity=at
        )

    def trajectory(
        self,
        config: C | dict[str, Any] | Configuration,
        *,
        frm: F | None = None,
        to: F | None = None,
        step: F | None = None,
    ) -> list[R]:
        """Get the full trajectory of a configuration

        Parameters
        ----------
        config : C | dict[str, Any] | Configuration
            The config to query

        frm: F | None = None
            Start of the curve, defaults to the minimum fidelity

        to: F | None = None
            End of the curve, defaults to the maximum fidelity

        step: F | None = None
            Step size, defaults to benchmark standard (1 for epoch)

        Returns
        -------
        list[R]
            A list of the results for this config
        """
        if isinstance(config, Configuration):
            config = {**config}

        if isinstance(config, YAHPOConfig):
            config = config.dict()

        assert isinstance(config, dict), "I assume this should be the case by here"

        if self.fidelity_name in config:
            msg = "``trajectory(config, frm=..., to=..., step=...)`` not config"
            raise ValueError(msg)

        # Copy same config and insert fidelities for each
        queries: list[dict] = [
            {**config, self.fidelity_name: f}
            for f in self.iter_fidelities(frm=frm, to=to, step=step)
        ]

        if self.task_id is not None:
            for q in queries:
                q["OpenML_task_id"] = self.task_id

        results = self.bench.objective_function(queries, seed=self.seed)

        return [
            self.Result.from_dict(
                config=self.Config(**config),  # Same config for each
                result=r,
                fidelity=q[self.fidelity_name],
            )
            for r, q in zip(results, queries)  # We need to loop over q's for fidelity
        ]

    @property
    def space(self) -> ConfigurationSpace:
        """The ConfigurationSpace for a YAHPO benchmark"""
        return self._configspace
