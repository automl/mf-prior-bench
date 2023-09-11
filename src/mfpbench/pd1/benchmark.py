from __future__ import annotations

import warnings
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Mapping, TypeVar
from typing_extensions import override

import numpy as np
import pandas as pd

from mfpbench.benchmark import Benchmark
from mfpbench.config import Config
from mfpbench.result import Result
from mfpbench.setup_benchmark import PD1Source

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace
    from xgboost import XGBRegressor


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class PD1Config(Config):
    """The config for PD1."""

    lr_decay_factor: float
    lr_initial: float
    lr_power: float
    opt_momentum: float


C = TypeVar("C", bound=PD1Config)


@dataclass(frozen=True)  # type: ignore[misc]
class PD1Result(Result[PD1Config, int]):
    valid_error_rate: float  # (0, 1)
    train_cost: float  #

    @property
    def score(self) -> float:
        """The score of interest."""
        return 1 - self.valid_error_rate

    @property
    def error(self) -> float:
        """The error of interest."""
        return self.valid_error_rate

    @property
    def val_score(self) -> float:
        """The score on the validation set."""
        return 1 - self.valid_error_rate

    @property
    def val_error(self) -> float:
        """The error on the validation set."""
        return self.valid_error_rate

    @property
    def cost(self) -> float:
        """The train cost of the model (asssumed to be seconds).

        Please double check with YAHPO.
        """
        return self.train_cost


@dataclass(frozen=True)  # type: ignore[misc]
class PD1ResultSimple(PD1Result):
    """Used for all PD1 benchmarks, except imagenet, lm1b, translate_wmt, uniref50."""

    test_error_rate: float = np.inf

    @property
    def test_score(self) -> float:
        """The score on the test set."""
        return 1 - self.test_error_rate

    @property
    def test_error(self) -> float:
        """The error on the test set."""
        return self.test_error_rate


@dataclass(frozen=True)
class PD1ResultTransformer(PD1Result):
    """Imagenet, lm1b, translate_wmt, uniref50, cifar100 contains no test error."""

    @property
    def test_score(self) -> float:
        """The score on the test set."""
        warnings.warn(
            "Using valid error rate as there is no test error rate",
            UserWarning,
            stacklevel=2,
        )
        return 1 - self.valid_error_rate

    @property
    def test_error(self) -> float:
        """The error on the test set."""
        warnings.warn(
            "Using valid error rate as there is no test error rate",
            UserWarning,
            stacklevel=2,
        )
        return self.valid_error_rate


R = TypeVar("R", PD1ResultTransformer, PD1ResultSimple)


class PD1Benchmark(Benchmark[C, R, int]):
    pd1_dataset: ClassVar[str]
    """The dataset that this benchmark uses."""

    pd1_model: ClassVar[str]
    """The model that this benchmark uses."""

    pd1_batchsize: ClassVar[int]
    """The batchsize that this benchmark uses."""

    pd1_metrics: ClassVar[tuple[str, ...]]
    """The metrics that are available for this benchmark."""

    Config: type[C]
    """The config type for this benchmark."""

    Result: type[R]
    """The result type for this benchmark."""

    has_conditionals = False

    def __init__(
        self,
        *,
        datadir: str | Path | None = None,
        seed: int | None = None,
        prior: str | Path | C | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
    ):
        """Create a PD1 Benchmark.

        Args:
            datadir: Path to the data directory
            seed: The seed to use for the space
            prior: Any prior to use for the benchmark
            perturb_prior: Whether to perturb the prior. If specified, this
                is interpreted as the std of a normal from which to perturb
                numerical hyperparameters of the prior, and the raw probability
                of swapping a categorical value.
        """
        cls = self.__class__
        space = cls._create_space(seed=seed)
        name = f"{cls.pd1_dataset}-{cls.pd1_model}-{cls.pd1_batchsize}"
        if datadir is None:
            datadir = PD1Source.default_location()

        datadir = Path(datadir) if isinstance(datadir, str) else datadir
        if not datadir.exists():
            raise FileNotFoundError(
                f"Can't find folder at {datadir}."
                f"\n`python -m mfpbench download --status --data-dir {datadir.parent}`",
            )
        self._surrogates: dict[str, XGBRegressor] | None = None
        self.datadir = datadir

        super().__init__(
            seed=seed,
            name=name,
            prior=prior,
            perturb_prior=perturb_prior,
            space=space,
        )

    def load(self) -> None:
        """Load the benchmark."""
        _ = self.surrogates  # Call up the surrogate into memory

    @property
    def surrogates(self) -> dict[str, XGBRegressor]:
        """The surrogates for this benchmark, one per metric."""
        if self._surrogates is None:
            from xgboost import XGBRegressor

            self._surrogates = {}
            for metric, path in self.surrogate_paths.items():
                if not path.exists():
                    raise FileNotFoundError(
                        f"Can't find surrogate at {path}."
                        "\n`python -m mfpbench download --status --data-dir "
                        f" {self.datadir.parent}",
                    )
                model = XGBRegressor()
                model.load_model(path)
                self._surrogates[metric] = model

        return self._surrogates

    @property
    def surrogate_dir(self) -> Path:
        """The directory where the surrogates are stored."""
        return self.datadir / "surrogates"

    @property
    def surrogate_paths(self) -> dict[str, Path]:
        """The paths to the surrogates."""
        return {
            metric: self.surrogate_dir / f"{self.name}-{metric}.json"
            for metric in self.pd1_metrics
        }

    @override
    def _objective_function(self, config: C, at: int) -> R:
        return self._results_for(config, fidelities=[at])[0]

    @override
    def _trajectory(self, config: C, *, frm: int, to: int, step: int) -> list[R]:
        return self._results_for(config, fidelities=self.iter_fidelities(frm, to, step))

    def _results_for(self, config: C, fidelities: Iterable[int]) -> list[R]:
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

    @classmethod
    @abstractmethod
    def _create_space(cls, seed: int | None = None) -> ConfigurationSpace:
        ...
