from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from mfpbench.config import TabularConfig
from mfpbench.metric import Metric
from mfpbench.result import Result
from mfpbench.setup_benchmark import PD1TabularSource  # TODO
from mfpbench.tabular import TabularBenchmark

from mfpbench.taskset_tabular.processing.process import PAPER_TASK_FAMILIES


def _get_raw_taskset_space(
    name: str,
    seed: int | None = None,
    *,
    optimizer: str | None = "adam8p",
) -> ConfigurationSpace:

    cs = ConfigurationSpace(name=name, seed=seed)
    cs.add_hyperparameters(
        [
            UniformFloatHyperparameter(
                "learning_rate",
                lower=1.026942e-08,
                upper=9.682791,
                log=True,
            )
        ]
    )
    if optimizer in ["adam4p", "adam6p", "adam8p"]:
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "beta1",
                    lower=4.807830e-04,
                    upper=0.9999,
                    log=False,
                ),
                UniformFloatHyperparameter(
                    "beta2",
                    lower=1.831740e-03,
                    upper=0.999999,
                    log=False,
                ),
                UniformFloatHyperparameter(
                    "epsilon",
                    lower=1.046320e-10,
                    upper=975.014812,
                    log=True,
                ),
            ]
        )
    if optimizer in ["adam6p", "adam8p"]:
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "l1",
                    lower=1.007364e-08,
                    upper=9.630265,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "l2",
                    lower=1.006209e-08,
                    upper=9.314683,
                    log=True,
                ),
            ]
        )
    if optimizer in ["adam8p"]:
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "linear_decay",
                    lower=1.002723e-07,
                    upper=0.000100,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "exponential_decay",
                    lower=1.003401e-06,
                    upper=0.000990,
                    log=True,
                ),
            ]
        )
    return cs


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TaskSetTabularConfig_1p(TabularConfig):
    learning_rate: float
    

@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TaskSetTabularConfig_4p(TabularConfig):
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TaskSetTabularConfig_6p(TabularConfig):
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    l1: float
    l2: float


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TaskSetTabularConfig_8p(TabularConfig):
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    l1: float
    l2: float
    linear_decay: float
    exponential_decay: float


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResults(Result[
    TaskSetTabularConfig_8p,
    int
]):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(0, np.inf)),
        "valid_loss": Metric(minimize=True, bounds=(0, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(0, np.inf)),
        "train_cost": Metric(minimize=True, bounds=(0, np.inf)),
    }
    default_value_metric: ClassVar[str] = "valid_loss"
    default_value_metric_test: ClassVar[str] = "test_loss"
    default_cost_metric: ClassVar[str] = "train_cost"

    train_loss: Metric.Value
    valid_loss: Metric.Value
    test_loss: Metric.Value
    train_cost: Metric.Value


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResults_6p(TaskSetTabularResults, Result[
    TaskSetTabularConfig_6p,
    int
]):
    pass


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResults_4p(TaskSetTabularResults, Result[
    TaskSetTabularConfig_4p,
    int
]):
    pass


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResults_1p(TaskSetTabularResults, Result[
    TaskSetTabularConfig_1p,
    int
]):
    pass


class TaskSetTabularBenchmark(TabularBenchmark):
    problem_types: ClassVar[tuple[str, ...]] = [
        "FixedTextRNNClassification"
    ]

    datasets: ClassVar[tuple[str, ...]] = [
        "imdb_patch32",
        "imdb_patch128",
    ]

    architectures: ClassVar[tuple[str, ...]] = [
        "LSTM128_avg",
        "GRU128",
        "GRU64_avg",
        "IRNN64_relu_avg",
        "LSTM128_E128",
        "VRNN128_tanh",
        "VRNN64_relu_avg",
        "VRNN64_tanh_avg",
    ]
    
    batch_sizes: ClassVar[tuple[str, ...]] = [
        "bs64",
        "bs128",
    ]

    optimizers: ClassVar[tuple[str, ...]] = [
        "adam1p",
        "adam4p",
        "adam6p",
        "adam8p",
    ]

    def __init__(
        self,
        dataset: str,
        architecture: str,
        batch_size: int,
        optimizer: str,
        problem_type: str = "FixedTextRNNClassification",
        datadir: str | Path | None = None,
        *,
        remove_constants: bool = False,
        seed: int | None = None,
        prior: str | Path | TaskSetTabularConfig_8p | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
        value_metric: str | None = None,
        value_metric_test: str | None = None,
        cost_metric: str | None = None,
    ) -> None:
        cls = self.__class__
        if dataset not in cls.datasets:
            raise ValueError(f"Unknown task {dataset}, must be one of {cls.datasets}")
        if architecture not in cls.architectures:
            raise ValueError(f"Unknown task {architecture}, must be one of {cls.architectures}")
        if batch_size not in cls.batch_sizes:
            raise ValueError(f"Unknown task {batch_size}, must be one of {cls.batch_sizes}")
        if optimizer not in cls.optimizers:
            raise ValueError(f"Unknown task {optimizer}, must be one of {cls.optimizers}")

        bench_name = f"{problem_type}_{dataset}_{architecture}_{batch_size}_{optimizer}"

        if datadir is None:
            datadir = PD1TabularSource.default_location()
    
        table_path = Path(datadir) / f"{bench_name}.parquet"
        print(table_path)
        if not table_path.exists():
            raise FileNotFoundError(
                f"Could not find table {table_path}."
                f"`python -m mfpbench download --status --data-dir {datadir}",
            )

        # Reading table
        table = pd.read_parquet(table_path)

        space = _get_raw_taskset_space(
            name=bench_name,
            seed=seed,
            optimizer=optimizer,
        )

        def get_config_type():
            if "1p" in optimizer:
                return TaskSetTabularConfig_1p
            elif "4p" in optimizer:
                return TaskSetTabularConfig_4p
            elif "6p" in optimizer:
                return TaskSetTabularConfig_6p
            elif "8p" in optimizer:
                return TaskSetTabularConfig_8p
            else:
                raise ValueError("Cannot recognize optimizer!")

        super().__init__(
            table=table,  # type: ignore
            name=bench_name,
            id_key="id",
            fidelity_key="epoch",
            result_type=TaskSetTabularResults,
            config_type=get_config_type(),
            info_keys=[],
            value_metric=value_metric,
            value_metric_test=value_metric_test,
            cost_metric=cost_metric,
            space=space,
            seed=seed,
            prior=prior,
            perturb_prior=perturb_prior,
        )
