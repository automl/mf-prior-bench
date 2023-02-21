from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

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

    yahpo_gym.local_config.init_config(data_path=str(path))
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

    # Name of hp used to indicate task
    _task_id_name: str | None = None

    # Any hyperparameters that should be forcefully deleted from the space
    # but have default values filled in
    _forced_hps: dict[str, int | float | str] | None = None
    # Any replacements that need to be done in hyperparameters
    # [(dataclass_version, dict_version)]
    _replacements_hps: list[tuple[str, str]] | None = None

    def __init__(
        self,
        task_id: str,
        *,
        datadir: str | Path | None = None,
        seed: int | None = None,
        prior: str | Path | C | dict[str, Any] | Configuration | None = None,
        perturb_prior: float | None = None,
    ):
        """Initialize a Yahpo Benchmark.

        Parameters
        ----------
        task_id: str
            The task id to choose from, see cls.instances.

        datadir : str | Path | None = None
            The path to where mfpbench stores it data. If left to default (None), will
            use the `_default_download_dir = ./data/yahpo-gym-data`.

        seed : int | None = None
            The seed for the benchmark instance

        prior: str | Path | YahpoConfig | None = None
            The prior to use for the benchmark.
            * if str - A preset
            * if Path - path to a file
            * if dict, Config, Configuration - A config

        perturb_prior: float | None = None
            If given, will perturb the prior by this amount. Only used if `prior` is
            given as a config.
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

        # Needs to be set before the call to super
        self.task_id = task_id

        super().__init__(seed=seed, prior=prior, perturb_prior=perturb_prior)
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

        bench = yahpo_gym.BenchmarkSet(self.name, instance=task_id, multithread=False)

        # These can have one or two fidelities
        # NOTE: seed is allowed to be int | None
        space = bench.get_opt_space(
            drop_fidelity_params=True,
            seed=seed,  # type: ignore
        )

        if self._task_id_name is not None:
            if self.has_conditionals:
                raise NotImplementedError(
                    f"{self.name} has conditionals, can't remove task_id from space"
                )

            space = remove_hyperparameter(self._task_id_name, space)

        if self._forced_hps is not None:
            names = space.get_hyperparameter_names()
            for key in self._forced_hps:
                if key in names:
                    if self.has_conditionals:
                        msg = (
                            f"{self.name} has conditionals,"
                            " can't remove task_id from space"
                        )
                        raise NotImplementedError(msg)

                    space = remove_hyperparameter(key, space)

        self._bench: yahpo_gym.BenchmarkSet | None = None
        self.datadir = datadir
        self._configspace = space

        if self.prior is not None:
            if self.perturb_prior is not None:
                self.prior = self.prior.perturb(
                    self._configspace,
                    seed=self.seed,
                    std=self.perturb_prior,
                    categorical_swap_chance=0,  # TODO
                )

            self.prior.set_as_default_prior(self._configspace)

    @property
    def bench(self) -> yahpo_gym.BenchmarkSet:
        if self._bench is None:
            bench = yahpo_gym.BenchmarkSet(
                self.name,
                instance=self.task_id,
                multithread=False,
            )
            self._bench = bench
        return self._bench

    def load(self) -> None:
        _ = self.bench

    @property
    def basename(self) -> str:
        return f"{self.name}-{self.task_id}"

    def query(
        self,
        config: C | dict[str, Any] | Configuration,
        at: F | None = None,
        *,
        argmax: bool = False,
    ) -> R:
        """Query the results for a config.

        Parameters
        ----------
        config : C | dict[str, Any] | Configuration
            The config to query

        at : F | None = None
            The fidelity at which to query, defaults to None which means *maximum*

        argmax: bool = False
            Whether to return the argmax up to the point `at`. Will be slower as it
            has to get the entire trajectory. Uses the corresponding Result's score.

        Returns
        -------
        R
            The result for the config at the given epoch
        """
        at = at if at is not None else self.end
        assert self.start <= at <= self.end

        if argmax:
            return max(self.trajectory(config, to=at), key=lambda r: r.score)

        if isinstance(config, (Configuration, dict)):
            config = self.Config.from_dict(config)

        assert isinstance(config, self.Config)

        query = config.dict()

        if self._forced_hps is not None:
            query.update(self._forced_hps)

        if self.task_id is not None and self._task_id_name is not None:
            query[self._task_id_name] = self.task_id

        query[self.fidelity_name] = at

        # NOTE: seed is allowed to be int | None
        results: list[dict] = self.bench.objective_function(
            query,
            seed=self.seed,  # type: ignore
        )
        result = results[0]

        return self.Result.from_dict(
            config=config,
            result=result,
            fidelity=at,
        )

    def trajectory(
        self,
        config: C | dict[str, Any] | Configuration,
        *,
        frm: F | None = None,
        to: F | None = None,
        step: F | None = None,
    ) -> list[R]:
        """Get the full trajectory of a configuration.

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
        if isinstance(config, (Configuration, dict)):
            config = self.Config.from_dict(config)

        assert isinstance(config, self.Config)

        query = config.dict()

        if self._forced_hps is not None:
            query.update(self._forced_hps)

        if self.task_id is not None and self._task_id_name is not None:
            query[self._task_id_name] = self.task_id

        # Copy same config and insert fidelities for each
        queries: list[dict] = [
            {**query, self.fidelity_name: f}
            for f in self.iter_fidelities(frm=frm, to=to, step=step)
        ]

        # NOTE: seed is allowed to be int | None
        results: list[dict] = self.bench.objective_function(
            queries,
            seed=self.seed,  # type: ignore
        )

        return [
            self.Result.from_dict(
                config=config,
                result=result,
                fidelity=query[self.fidelity_name],
            )
            # We need to loop over q's for fidelity
            for result, query in zip(results, queries)
        ]

    @property
    def space(self) -> ConfigurationSpace:
        """The ConfigurationSpace for a YAHPO benchmark."""
        return self._configspace
