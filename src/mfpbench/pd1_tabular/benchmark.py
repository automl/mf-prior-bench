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

from mfpbench.pd1.benchmark import (
    PD1Config, PD1ResultSimple, PD1ResultTransformer, PD1Benchmark
)


def _get_raw_lcbench_space(
    name: str,
    seed: int | None = None,
    *,
    with_constants: bool | None = None,
) -> ConfigurationSpace:

    cs = ConfigurationSpace(name=name, seed=seed)
    cs.add_hyperparameters(
        [
            UniformFloatHyperparameter(
                "lr_decay_factor",
                lower=0.01,
                upper=0.99,
            ),
            UniformFloatHyperparameter(
                "lr_initial",
                lower=1.0e-5,
                upper=10,
                log=True,
            ),
            UniformFloatHyperparameter(
                "lr_power",
                lower=0.1,
                upper=2.0,
            ),
            UniformFloatHyperparameter(
                "opt_momentum",
                lower=1.0e-3,
                upper=1.0,
                log=True,
            ),
        ],
    )
    return cs


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class PD1TabularConfig(TabularConfig):
    lr_decay_factor: float
    lr_initial: float
    lr_power: float
    opt_momentum: float
    # not part of actual config space
    original_epoch: int


@dataclass(frozen=True)  # type: ignore[misc]
class PD1TabularResultSimple(PD1ResultSimple):
    pass


@dataclass(frozen=True)
class PD1TabularResultTransformer(PD1ResultTransformer):
    pass


class PD1TabularBenchmark(TabularBenchmark):
    datasets: ClassVar[tuple[str, ...]] = (
        "cifar10",
        "cifar100",
        "fashion_mnist",
        "imagenet",
        "lm1b",
        "mnist",
        "svhn_no_extra",
        "translate_wmt",
        "uniref50"
    )
    non_test_datasets: ClassVar[tuple[str, ...]] = (
        "imagenet",
        "lm1b",
        "translate_wmt",
        "uniref50"
    )

    models: ClassVar[tuple[str, ...]] = (
        "wide_resnet",
        "max_pooling_cnn",
        "simple_cnn",
        "resnet",
        "transformer",
        "xformer_translate"
    )

    batch_sizes: ClassVar[tuple[str, ...]] = (
        64,
        128,
        256,
        512,
        1024,
        2048
    )

    def __init__(
        self,
        dataset: str,
        model: str,
        batch_size: int,
        datadir: str | Path | None = None,
        *,
        remove_constants: bool = False,
        seed: int | None = None,
        prior: str | Path | PD1TabularConfig | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
        value_metric: str | None = None,
        value_metric_test: str | None = None,
        cost_metric: str | None = None,
    ) -> None:
        cls = self.__class__

        if dataset not in cls.datasets:
            raise ValueError(f"Unknown task {dataset}, must be one of {cls.datasets}")
        if model not in cls.models:
            raise ValueError(f"Unknown task {model}, must be one of {cls.models}")
        if batch_size not in cls.batch_sizes:
            raise ValueError(f"Unknown task {batch_size}, must be one of {cls.batch_sizes}")

        bench_name = f"{dataset}-{model}-{batch_size}"

        if datadir is None:
            datadir = PD1TabularSource.default_location()
        
        table_path = Path(datadir) / f"{bench_name}_tabular.parquet"
        if not table_path.exists():
            raise FileNotFoundError(
                f"Could not find table {table_path}."
                f"`python -m mfpbench download --status --data-dir {datadir}",
            )

        # Reading table
        table = pd.read_parquet(table_path)

        space = _get_raw_lcbench_space(
            name=bench_name,
            seed=seed,
            with_constants=not remove_constants,
        )

        super().__init__(
            table=table,  # type: ignore
            name=bench_name,
            id_key="id",
            fidelity_key="epoch",
            result_type=(
                PD1TabularResultTransformer
                if dataset in cls.non_test_datasets
                else PD1TabularResultSimple
            ),
            config_type=PD1TabularConfig,
            value_metric=value_metric,
            value_metric_test=value_metric_test,
            cost_metric=cost_metric,
            space=space,
            seed=seed,
            prior=prior,
            perturb_prior=perturb_prior,
        )