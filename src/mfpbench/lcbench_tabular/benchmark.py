from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from mfpbench.config import TabularConfig
from mfpbench.result import Result
from mfpbench.setup_benchmark import LCBenchTabularSource
from mfpbench.tabular import TabularBenchmark


def _get_raw_lcbench_space(
    name: str,
    seed: int | None = None,
    *,
    with_constants: bool = False,
) -> ConfigurationSpace:
    """Get the raw configuration space for lcbench tabular.

    !!! note

        This configuration space is the same across all tasks that lcbench tabular
        has.

    Args:
        name: The name for the space.
        seed: The seed to use.
        with_constants: Whether to include constants or not

    Returns:
        The configuration space.
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
        ],
    )

    if with_constants:
        cs.add_hyperparameters(
            [
                Constant("cosine_annealing_T_max", 50),
                Constant("cosine_annealing_eta_min", 0.0),
                Constant("normalization_strategy", "standardize"),
                Constant("optimizer", "sgd"),
                Constant("learning_rate_scheduler", "cosine_annealing"),
                Constant("network", "shapedmlpnet"),
                Constant("activation", "relu"),
                Constant("mlp_shape", "funnel"),
                Constant("imputation_strategy", "mean"),
            ],
        )
    return cs


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class LCBenchTabularConfig(TabularConfig):
    batch_size: int
    max_dropout: float
    max_units: int
    num_layers: int
    learning_rate: float
    momentum: float
    weight_decay: float
    # All of these are constant and hence optional
    loss: str | None = None  # This is the name of the loss function used, not a float
    imputation_strategy: str | None = None
    learning_rate_scheduler: str | None = None
    network: str | None = None
    normalization_strategy: str | None = None
    optimizer: str | None = None
    cosine_annealing_T_max: int | None = None
    cosine_annealing_eta_min: float | None = None
    activation: str | None = None
    mlp_shape: str | None = None


@dataclass(frozen=True)  # type: ignore[misc]
class LCBenchTabularResult(Result[LCBenchTabularConfig, int]):
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
        return self.val_score

    @property
    def error(self) -> float:
        """The error of interest."""
        return self.val_error

    @property
    def val_score(self) -> float:
        """The score on the validation set."""
        return self.val_accuracy / 100

    @property
    def val_error(self) -> float:
        """The error on the validation set."""
        return (100 - self.val_accuracy) / 100

    @property
    def test_score(self) -> float:
        """The score on the test set."""
        return self.test_accuracy / 100

    @property
    def test_error(self) -> float:
        """The error on the test set."""
        return (100 - self.test_accuracy) / 100

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

        # NOTE: Dropping of 0'th epoch
        # As the 0'th epoch is a completely untrained model, this is different
        # from 1st epoch where it is trained and it's score is somewhat representitive.
        # This is a benchmarking library for HPO and we do not want to include untrained
        # models nor have it be part of the fidelity range. For that reason, we drop
        # the 0'th epoch.
        drop_epoch = 0
        table = table.drop(index=drop_epoch, level="epoch")
        benchmark_task_name = f"lcbench_tabular-{task_id}"
        space = _get_raw_lcbench_space(
            name=f"lcbench_tabular-{task_id}",
            seed=seed,
            with_constants=not remove_constants,
        )

        # for tabular benchmarks, the `global` optima can be retrieved
        optimal_metrics = _get_optima(table)

        super().__init__(
            table=table,  # type: ignore
            name=benchmark_task_name,
            id_key="id",
            fidelity_key=cls.fidelity_name,
            result_keys=LCBenchTabularResult.names(),
            config_keys=LCBenchTabularConfig.names(),
            remove_constants=remove_constants,
            space=space,
            seed=seed,
            prior=prior,
            perturb_prior=perturb_prior,
            optimal_metrics=optimal_metrics
        )


def _get_optima(table: pd.DataFrame) -> dict:
    metrics = [
        ("val_accuracy", max),
        ("val_cross_entropy", min),
        ("val_balanced_accuracy", max),
        ("test_accuracy", max),
        ("test_cross_entropy", min),
        ("test_balanced_accuracy", max)
    ]
    optimal_metrics = {
        metric_name: metric_order(table.loc[:, metric_name])
        for i, (metric_name, metric_order) in enumerate(metrics)
    }
    return optimal_metrics
