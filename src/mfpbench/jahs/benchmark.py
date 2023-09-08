from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping
from typing_extensions import override

import numpy as np
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    Constant,
    UniformFloatHyperparameter,
)

from mfpbench.benchmark import Benchmark
from mfpbench.config import Config
from mfpbench.result import Result
from mfpbench.setup_benchmark import JAHSBenchSource
from mfpbench.util import rename

if TYPE_CHECKING:
    import jahs_bench


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class JAHSConfig(Config):
    """The config for JAHSBench, useful to have regardless of the configspace used.

    https://github.com/automl/jahs_bench_201/blob/main/jahs_bench/lib/core/configspace.py
    """

    # Not fidelities for our use case
    N: int
    W: int

    # Categoricals
    Op1: int
    Op2: int
    Op3: int
    Op4: int
    Op5: int
    Op6: int
    TrivialAugment: bool
    Activation: str
    Optimizer: str

    # Continuous Numericals
    Resolution: float
    LearningRate: float
    WeightDecay: float

    def validate(self) -> None:
        """Validate this config incase required."""
        # Just being explicit to catch bugs easily, we can remove later
        assert self.N in [1, 3, 5]
        assert self.W in [4, 8, 16]
        assert self.Op1 in [0, 1, 2, 3, 4, 5]
        assert self.Op2 in [0, 1, 2, 3, 4, 5]
        assert self.Op3 in [0, 1, 2, 3, 4, 5]
        assert self.Op4 in [0, 1, 2, 3, 4, 5]
        assert self.Op5 in [0, 1, 2, 3, 4, 5]
        assert self.Op6 in [0, 1, 2, 3, 4, 5]
        assert self.Resolution in [0.25, 0.5, 1.0]
        assert isinstance(self.TrivialAugment, bool)
        assert self.Activation in ["ReLU", "Hardswish", "Mish"]
        assert self.Optimizer in ["SGD"]
        assert 1e-3 <= self.LearningRate <= 1e0
        assert 1e-5 <= self.WeightDecay <= 1e-2


@dataclass(frozen=True)  # type: ignore[misc]
class JAHSResult(Result[JAHSConfig, int]):
    # Info
    # size: float  # remove
    # flops: float # remove
    # latency: float  # unit? remove
    runtime: float  # unit?

    # Scores (0 - 100)
    valid_acc: float
    test_acc: float
    # train_acc: float # remove

    @property
    def score(self) -> float:
        """The score of interest."""
        return self.valid_acc

    @property
    def error(self) -> float:
        """The error of interest."""
        return 100 - self.valid_acc

    @property
    def test_score(self) -> float:
        """The score on the test set."""
        return self.test_acc

    @property
    def test_error(self) -> float:
        """The error on the test set."""
        return 100 - self.test_acc

    @property
    def val_score(self) -> float:
        """The score on the validation set."""
        return self.valid_acc

    @property
    def val_error(self) -> float:
        """The error on the validation set."""
        return 100 - self.valid_acc

    @property
    def cost(self) -> float:
        """The time taken (assumed to be seconds)."""
        return self.runtime


class JAHSBenchmark(Benchmark[JAHSConfig, JAHSResult, int], ABC):
    Config = JAHSConfig
    Result = JAHSResult
    fidelity_name = "epoch"
    fidelity_range = (3, 200, 1)  # TODO: min budget plays a huge role in SH/HB algos

    task_ids: tuple[str, ...] = (
        "CIFAR10",
        "ColorectalHistology",
        "FashionMNIST",
    )
    """
    ```python exec="true" result="python"
    from mfpbench import JAHSBenchmark
    print(JAHSBenchmark.task_ids)
    ```
    """

    _result_renames: Mapping[str, str] = {
        "size_MB": "size",
        "FLOPS": "flops",
        "valid-acc": "valid_acc",
        "test-acc": "test_acc",
        "train-acc": "train_acc",
    }
    _result_metrics_active: tuple[str, ...] = ("valid-acc", "test-acc", "runtime")

    def __init__(
        self,
        task_id: Literal["CIFAR10", "ColorectalHistology", "FashionMNIST"],
        *,
        datadir: str | Path | None = None,
        seed: int | None = None,
        prior: str | Path | JAHSConfig | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
    ):
        """Initialize the benchmark.

        Args:
            task_id: The specific task to use.
            datadir: The path to where mfpbench stores it data. If left to `None`,
                will use `#!python _default_download_dir = "./data/jahs-bench-data"`.
            seed: The seed to give this benchmark instance
            prior: The prior to use for the benchmark.

                * if `str` - A preset
                * if `Path` - path to a file
                * if `dict`, Config, Configuration - A config
                * if `None` - Use the default if available

            perturb_prior: If given, will perturb the prior by this amount.
                Only used if `prior=` is given as a config.
        """
        cls = self.__class__
        if datadir is None:
            datadir = JAHSBenchSource.default_location()

        datadir = Path(datadir)

        if not datadir.exists():
            raise FileNotFoundError(
                f"Can't find folder at {datadir}."
                f"\n`python -m mfpbench download --status --data-dir {datadir}`",
            )

        # Loaded on demand with `@property`
        self._bench: jahs_bench.Benchmark | None = None
        self.datadir = datadir
        self.task_id = task_id

        name = f"jahs_{task_id}"
        super().__init__(
            seed=seed,
            name=name,
            space=cls._jahs_configspace(name=name, seed=seed),
            prior=prior,
            perturb_prior=perturb_prior,
        )

    # explicit overwrite
    def load(self) -> None:
        """Pre-load JAHS XGBoost model before querying the first time."""
        # Access the property
        _ = self.bench

    @property
    def bench(self) -> jahs_bench.Benchmark:
        """The underlying benchmark used."""
        if not self._bench:
            try:
                import jahs_bench
            except ImportError as e:
                raise ImportError(
                    "jahs-bench not installed, please install it with "
                    "`pip install jahs-bench`",
                ) from e

            tasks = {
                "CIFAR10": jahs_bench.BenchmarkTasks.CIFAR10,
                "ColorectalHistology": jahs_bench.BenchmarkTasks.ColorectalHistology,
                "FashionMNIST": jahs_bench.BenchmarkTasks.FashionMNIST,
            }
            task = tasks.get(self.task_id, None)
            if task is None:
                raise ValueError(
                    f"Unknown task {self.task_id}, must be in {list(tasks.keys())}",
                )

            self._bench = jahs_bench.Benchmark(
                task=self.task_id,
                save_dir=self.datadir,
                download=False,
                metrics=self._result_metrics_active,
            )

        return self._bench

    @override
    def _objective_function(self, config: JAHSConfig, at: int) -> JAHSResult:
        query = config.dict()

        results = self.bench.__call__(query, nepochs=at)
        result = results[at]

        return self.Result.from_dict(
            config=config,
            result=rename(result, keys=self._result_renames),
            fidelity=at,
        )

    @override
    def _trajectory(
        self,
        config: JAHSConfig,
        *,
        frm: int,
        to: int,
        step: int,
    ) -> list[JAHSResult]:
        query = config.dict()

        try:
            results = self.bench.__call__(query, nepochs=to, full_trajectory=True)
        except TypeError:
            # See: https://github.com/automl/jahs_bench_201/issues/5
            results = {
                f: self.bench.__call__(query, nepochs=f)[f]
                for f in self.iter_fidelities(frm=frm, to=to, step=step)
            }

        return [
            self.Result.from_dict(
                config=config,
                fidelity=i,
                result=rename(results[i], keys=self._result_renames),
            )
            for i in self.iter_fidelities(frm=frm, to=to, step=step)
        ]

    @classmethod
    def _jahs_configspace(
        cls,
        name: str = "jahs_bench_config_space",
        seed: int | np.random.RandomState | None = None,
    ) -> ConfigurationSpace:
        """The configuration space for all datasets in JAHSBench.

        Args:
            name: The name to give to the config space.
            seed: The seed to use for the config space

        Returns:
            The space
        """
        # Copied from https://github.com/automl/jahs_bench_201/blob/c1e92dd92a0c4906575c4e3e4ee9e7420efca5f1/jahs_bench/lib/core/configspace.py#L4  # noqa: E501
        # See for why we copy: https://github.com/automl/jahs_bench_201/issues/4
        if isinstance(seed, np.random.RandomState):
            seed = seed.tomaxint()

        try:
            from jahs_bench.lib.core.constants import Activations
        except ImportError as e:
            raise ImportError(
                "jahs-bench not installed, please install it with "
                "`pip install jahs-bench`",
            ) from e

        space = ConfigurationSpace(name=name, seed=seed)
        space.add_hyperparameters(
            [
                Constant(
                    "N",
                    # sequence=[1, 3, 5],
                    value=5,  # This is the value for NB201
                ),
                Constant(
                    "W",
                    # sequence=[4, 8, 16],
                    value=16,  # This is the value for NB201
                ),
                CategoricalHyperparameter(
                    "Op1",
                    choices=list(range(5)),
                    default_value=0,
                ),
                CategoricalHyperparameter(
                    "Op2",
                    choices=list(range(5)),
                    default_value=0,
                ),
                CategoricalHyperparameter(
                    "Op3",
                    choices=list(range(5)),
                    default_value=0,
                ),
                CategoricalHyperparameter(
                    "Op4",
                    choices=list(range(5)),
                    default_value=0,
                ),
                CategoricalHyperparameter(
                    "Op5",
                    choices=list(range(5)),
                    default_value=0,
                ),
                CategoricalHyperparameter(
                    "Op6",
                    choices=list(range(5)),
                    default_value=0,
                ),
                # OrdinalHyperparameter(
                #     "Resolution",
                #     sequence=[0.25, 0.5, 1.0],
                #     default_value=1.0,
                # ),
                Constant("Resolution", value=1.0),
                CategoricalHyperparameter(
                    "TrivialAugment",
                    choices=[True, False],
                    default_value=False,
                ),
                CategoricalHyperparameter(
                    "Activation",
                    choices=list(Activations.__members__.keys()),
                    default_value="ReLU",
                ),
            ],
        )

        # Add Optimizer related HyperParamters
        optimizers = Constant("Optimizer", value="SGD")
        lr = UniformFloatHyperparameter(
            "LearningRate",
            lower=1e-3,
            upper=1e0,
            default_value=1e-1,
            log=True,
        )
        weight_decay = UniformFloatHyperparameter(
            "WeightDecay",
            lower=1e-5,
            upper=1e-2,
            default_value=5e-4,
            log=True,
        )

        space.add_hyperparameters([optimizers, lr, weight_decay])
        return space
