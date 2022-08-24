from __future__ import annotations

from typing import Any, TypeVar

from pathlib import Path

import numpy as np
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

        self.datadir = Path(datadir) if isinstance(datadir, str) else datadir
        if not self.datadir.exists():
            raise FileNotFoundError(
                f"Can't find folder at {self.datadir}, have you run\n"
                f"`python -m mfpbench.download --data-dir {self.datadir.parent}`"
            )
        _ensure_yahpo_config_set(self.datadir)

        bench = yahpo_gym.BenchmarkSet(self.name, instance=task_id)

        # These can have one or two fidelities
        space = bench.get_opt_space(drop_fidelity_params=True, seed=seed)
        remove_hyperparameter("OpenML_task_id", space)

        self.bench = bench
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

        config[self.fidelity_name] = at
        result = self.bench.objective_function(configuration=config, seed=self.seed)

        del config[self.fidelity_name]

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
            Start of the curve, should default to the start

        to: F | None = None
            End of the curve, should default to the total

        step: F | None = None
            Step size to traverse, should default to ``cls.fidelity_range`` which
            is ``(start, stop, step)``

        Returns
        -------
        list[R]
            A list of the results for this config
        """
        frm = frm if frm is not None else self.start
        to = to if to is not None else self.end
        step = step if step is not None else self.step
        assert self.start <= frm <= to <= self.end

        if isinstance(config, Configuration):
            config = {**config}

        if isinstance(config, YAHPOConfig):
            config = config.dict()

        assert isinstance(config, dict), "I assume this should be the case by here"

        fidelities = np.arange(frm, to + step, step)

        # Note: Clamping floats on arange
        #
        #   There's an annoying detail about floats here, essentially we could over
        #   (frm=0.03, to + step = 1+ .05, step=0.5) -> [0.03, 0.08, ..., 1.03]
        #   We just clamp this to the last fidelity
        #
        #   This does not effect ints
        if isinstance(step, float) and fidelities[-1] >= self.end:
            fidelities[-1] = self.end

        # Copy same config and insert fidelities for each
        queries = [{**config, self.fidelity_name: f} for f in fidelities]
        results = self.bench.objective_function(queries)

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
        return self._configspace
