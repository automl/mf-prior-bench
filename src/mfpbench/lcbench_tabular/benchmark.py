from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import pandas as pd
from ConfigSpace import (
    ConfigurationSpace, Constant, UniformFloatHyperparameter, UniformIntegerHyperparameter
)

from mfpbench.config import TabularConfig
from mfpbench.result import Result
from mfpbench.setup_benchmark import LCBenchTabularSource
from mfpbench.tabular import TabularBenchmark


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class LCBenchTabularConfig(TabularConfig):
    batch_size: int
    # loss: str
    imputation_strategy: str
    learning_rate_scheduler: str
    network: str
    max_dropout: float
    normalization_strategy: str
    optimizer: str
    cosine_annealing_T_max: int
    cosine_annealing_eta_min: float
    activation: str
    max_units: int
    mlp_shape: str
    num_layers: int
    learning_rate: float
    momentum: float
    weight_decay: float


@dataclass(frozen=True)  # type: ignore[misc]
class LCBenchTabularResult(Result[LCBenchTabularConfig, int]):
    loss: float
    time: float
    val_accuracy: float
    val_cross_entropy: float
    val_balanced_accuracy: float
    test_accuracy: float
    test_cross_entropy: float
    test_balanced_accuracy: float

    @property
    def score(self) -> float:
        """The score of interest."""
        return self.val_accuracy

    @property
    def error(self) -> float:
        """The error of interest."""
        return 1 - self.val_error

    @property
    def val_score(self) -> float:
        """The score on the validation set."""
        return self.val_accuracy

    @property
    def val_error(self) -> float:
        """The error on the validation set."""
        return 1 - self.val_accuracy

    @property
    def test_score(self) -> float:
        """The score on the test set."""
        return self.test_accuracy

    @property
    def test_error(self) -> float:
        """The error on the test set."""
        return 1 - self.test_accuracy

    @property
    def cost(self) -> float:
        """The time to train the configuration (assumed to be seconds)."""
        return self.time


class LCBenchTabularBenchmark(TabularBenchmark):
    Config = LCBenchTabularConfig
    Result = LCBenchTabularResult
    fidelity_name: str = "epoch"

    task_ids: ClassVar[tuple[str, ...]] = (
        "adult",
        "airlines",
        "albert",
        "Amazon_employee_access",
        "APSFailure",
        "Australian",
        "bank-marketing",
        "blood-transfusion-service-center",
        "car",
        "christine",
        "cnae-9",
        "connect-4",
        "covertype",
        "credit-g",
        "dionis",
        "fabert",
        "Fashion-MNIST",
        "helena",
        "higgs",
        "jannis",
        "jasmine",
        "jungle_chess_2pcs_raw_endgame_complete",
        "kc1",
        "KDDCup09_appetency",
        "kr-vs-kp",
        "mfeat-factors",
        "MiniBooNE",
        "nomao",
        "numerai28.6",
        "phoneme",
        "segment",
        "shuttle",
        "sylvine",
        "vehicle",
        "volkert",
    )
    """
    ```python exec="true" result="python"
    from mfpbench import LCBenchTabularBenchmark
    print(LCBenchTabularBenchmark.task_ids)
    ```
    """

    def __init__(
        self,
        task_id: str,
        datadir: str | Path | None = None,
        *,
        remove_constants: bool = False,
        seed: int | None = None,
        prior: str | Path | LCBenchTabularConfig | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
    ) -> None:
        """Initialize the benchmark.

        Args:
            task_id: The task to benchmark on.
            datadir: The directory to look for the data in. If `None`, uses the default
                download directory.
            remove_constants: Whether to remove constant config columns from the data or
                not.
            seed: The seed to use.
            prior: The prior to use for the benchmark. If None, no prior is used.
                If a str, will check the local location first for a prior
                specific for this benchmark, otherwise assumes it to be a Path.
                If a Path, will load the prior from the path.
                If a Mapping, will be used directly.
            perturb_prior: If not None, will perturb the prior by this amount.
                For numericals, this is interpreted as the standard deviation of a
                normal distribution while for categoricals, this is interpreted
                as the probability of swapping the value for a random one.
        """
        cls = self.__class__
        if task_id not in cls.task_ids:
            raise ValueError(f"Unknown task {task_id}, must be one of {cls.task_ids}")

        if datadir is None:
            datadir = LCBenchTabularSource.default_location()

        table_path = Path(datadir) / f"{task_id}.parquet"
        if not table_path.exists():
            raise FileNotFoundError(
                f"Could not find table {table_path}."
                f"`python -m mfpbench download --status --data-dir {datadir}",
            )

        self.task_id = task_id
        self.datadir = Path(datadir) if isinstance(datadir, str) else datadir

        table = pd.read_parquet(table_path)
        super().__init__(
            table=table,
            name=f"lcbench_tabular-{task_id}",
            config_name="id",
            fidelity_name=cls.fidelity_name,
            result_keys=LCBenchTabularResult.names(),
            config_keys=LCBenchTabularConfig.names(),
            remove_constants=remove_constants,
            seed=seed,
            prior=prior,
            perturb_prior=perturb_prior,
        )

    @property
    def fidelity_range(self) -> tuple[int, int, int]:
        return (1, 51, 1)
        
    def get_raw_space(self, name: int | None = None, seed: int | None = None) -> ConfigurationSpace:
        """Create the configuration space for the benchmark.

        Args:
            name: The name for the configuration space.
            seed: The seed to use for the configuration space.

        Returns:
            The configuration space for the benchmark.
        """
        # obtained from https://github.com/automl/lcbench#dataset-overview
        cs = ConfigurationSpace(name=name, seed=seed)
        cs.add_hyperparameters(
            [
                UniformIntegerHyperparameter(
                    "batch_size",
                    lower=16,
                    upper=512,
                    log=True,
                    default_value=128,  # approximately log-spaced middle of range
                ),
                UniformFloatHyperparameter(
                    "learning_rate",
                    lower=1.0e-4,
                    upper=1.0e-1,
                    log=True,
                    default_value=1.0e-3,  # popular choice of LR
                ),
                UniformFloatHyperparameter(
                    "momentum",
                    lower=0.1,
                    upper=0.99,
                    log=False,
                    default_value=0.9,  # popular choice, also not on the boundary
                ),
                UniformFloatHyperparameter(
                    "weight_decay",
                    lower=1.0e-5,
                    upper=1.0e-1,
                    log=False,
                    default_value=1.0e-2,  # reasonable default
                ),
                UniformIntegerHyperparameter(
                    "num_layers",
                    lower=1,
                    upper=5,
                    log=False,
                    default_value=3,  # middle of range
                ),
                UniformIntegerHyperparameter(
                    "max_units",
                    lower=64,
                    upper=1024,
                    log=True,
                    default_value=256,  # approximately log-spaced middle of range
                ),
                UniformFloatHyperparameter(
                    "max_dropout",
                    lower=0,
                    upper=1,
                    log=False,
                    default_value=0.2,  # reasonable default
                ),
                *self._get_constant_hyperparameters(),
            ],
        )
        return cs
    
    def _get_constant_hyperparameters(self) -> list:
        constants = [
            Constant("cosine_annealing_T_max", 50),
            Constant("cosine_annealing_eta_min", 0.0),
            Constant("normalization_strategy", "standardize"),
            Constant("optimizer", "sgd"),
            Constant("learning_rate_scheduler", "cosine_annealing"),
            Constant("network", "shapedmlpnet"),
            Constant("activation", "relu"),
            Constant("mlp_shape", "funnel"),
            Constant("imputation_strategy", "mean"),
        ]
        return constants
