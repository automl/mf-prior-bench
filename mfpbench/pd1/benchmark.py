from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace
from xgboost import XGBRegressor

from mfpbench.benchmark import Benchmark
from mfpbench.download import DATAROOT
from mfpbench.pd1.config import PD1Config
from mfpbench.pd1.result import PD1ResultSimple, PD1ResultTransformer

C = TypeVar("C", bound=PD1Config)
R = TypeVar("R", PD1ResultTransformer, PD1ResultSimple)


class PD1Benchmark(Benchmark[C, R, int]):

    fidelity_name: str

    Config: type[C]
    Result: type[R]

    dataset: str
    model: str
    batchsize: int
    metrics: tuple[str, ...]

    fidelity_range: tuple[int, int, int]  # = (1, 200, 1)

    # Where the data for pd1 data should be located relative to the path given
    _default_download_dir: Path = DATAROOT / "pd1-data"
    _default_surrogate_dir: Path = DATAROOT / "pd1-data" / "surrogates"

    has_conditionals = False

    def __init__(
        self,
        *,
        datadir: str | Path | None = None,
        seed: int | None = None,
        prior: str | Path | C | dict[str, Any] | Configuration | None = None,
        perturb_prior: float | None = None,
    ):
        super().__init__(seed=seed, prior=prior, perturb_prior=perturb_prior)

        if datadir is None:
            datadir = PD1Benchmark._default_download_dir

        self.datadir = Path(datadir) if isinstance(datadir, str) else datadir
        if not self.datadir.exists():
            raise FileNotFoundError(
                f"Can't find folder at {self.datadir}, have you run\n"
                f"`python -m mfpbench.download --data-dir {self.datadir.parent}`"
            )

        self._configspace = self._create_space(seed=self.seed)

        if self.prior is not None:
            if self.perturb_prior is not None:
                self.prior = self.prior.perturb(
                    self._configspace,
                    seed=self.seed,
                    std=self.perturb_prior,
                    categorical_swap_chance=0,  # TODO
                )

            self.prior.set_as_default_prior(self._configspace)

        self._surrogates: dict[str, XGBRegressor] | None = None

    @property
    def basename(self) -> str:
        return self.dataset_name()

    def load(self) -> None:
        _ = self.surrogates  # Call up the surrogate into memory

    @property
    def surrogates(self) -> dict[str, XGBRegressor]:
        if self._surrogates is None:
            self._surrogates = {}
            for metric, path in self.surrogate_paths.items():
                if not path.exists():
                    raise FileNotFoundError(
                        f"Can't find surrogate at {path}, have you run\n"
                        f"`python -m mfpbench download`?"
                    )
                model = XGBRegressor()
                model.load_model(path)
                self._surrogates[metric] = model

        return self._surrogates

    @property
    def surrogate_dir(self) -> Path:
        return self.datadir / "surrogates"

    @property
    def surrogate_paths(self) -> dict[str, Path]:
        return {
            metric: self.surrogate_dir / f"{self.dataset_name()}-{metric}.json"
            for metric in self.metrics
        }

    @classmethod
    def dataset_name(cls) -> str:
        return f"{cls.dataset}-{cls.model}-{cls.batchsize}"

    def query(
        self,
        config: C | dict[str, Any] | Configuration,
        at: int | None = None,
        *,
        argmax: bool = False,
    ) -> R:
        """Query the results for a config.

        Parameters
        ----------
        config : C | dict[str, Any] | Configuration
            The config to query

        at : int | None = None
            The epoch at which to query at, defaults to max (200) if left as None

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

        if isinstance(config, Configuration):
            config = self.Config.from_dict({**config})  # type: ignore

        if isinstance(config, dict):
            config = self.Config.from_dict(config)

        assert isinstance(config, self.Config), f"Nope, config is {type(config)}"

        return self._results_for(config, fidelities=[at])[0]

    def trajectory(
        self,
        config: C | dict[str, Any] | Configuration,
        *,
        frm: int | None = None,
        to: int | None = None,
        step: int | None = None,
    ) -> list[R]:
        """Query the trajectory of a config as it ranges over a fidelity.

        Parameters
        ----------
        config : C | dict[str, Any] | Configuration
            The config to query

        frm: int | None = None
            Start of the curve, defaults to the minimum fidelity (1)

        to: int | None = None
            End of the curve, defaults to the maximum fidelity (200)

        step: int | None = None
            Step size, defaults to benchmark standard (1 for epoch)

        Returns
        -------
        list[R]
            The results over that trajectory
        """
        if isinstance(config, Configuration):
            config = self.Config.from_dict({**config})  # type: ignore

        if isinstance(config, dict):
            config = self.Config.from_dict(config)

        assert isinstance(config, self.Config), f"Nope, config is {type(config)}"

        fidelities = list(self.iter_fidelities(frm, to, step))

        return self._results_for(config, fidelities=fidelities)

    @property
    def space(self) -> ConfigurationSpace:
        return self._configspace

    @classmethod
    @abstractmethod
    def _create_space(cls, seed: int | None = None) -> ConfigurationSpace:
        ...

    def _results_for(self, config: C, fidelities: list[int]) -> list[R]:
        # Add the fidelities into the query and make a dataframe
        c = config.dict()
        queries = [{**c, self.fidelity_name: f} for f in fidelities]
        xs = pd.DataFrame(queries)

        # Predict the metric for everything in the dataframe
        features = xs.columns
        for metric, surrogate in self.surrogates.items():
            xs[metric] = surrogate.predict(xs[features])

        metrics = list(self.surrogates.keys())

        return [
            self.Result.from_dict(
                config=config,  # Our original config
                fidelity=r[self.fidelity_name],  # fidelity  # type: ignore
                result=r[metrics],  # Grab the metrics  # type: ignore
            )
            for _, r in xs.iterrows()
        ]
