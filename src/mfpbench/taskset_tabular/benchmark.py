from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, TypeVar

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

from mfpbench.config import TabularConfig
from mfpbench.metric import Metric
from mfpbench.result import Result
from mfpbench.tabular import TabularBenchmark


def _get_raw_taskset_space(
    name: str,
    seed: int | None = None,
    *,
    optimizer: str,
) -> ConfigurationSpace:
    cs = ConfigurationSpace(name=name, seed=seed)
    cs.add_hyperparameters(
        [
            UniformFloatHyperparameter(
                "learning_rate",
                lower=1.026942e-08,
                upper=9.682791,
                log=True,
            ),
        ],
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
            ],
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
            ],
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
            ],
        )
    return cs


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TaskSetTabularConfig(TabularConfig):
    pass


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TaskSetTabularConfig_1p(TaskSetTabularConfig):
    learning_rate: float


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TaskSetTabularConfig_4p(TaskSetTabularConfig):
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TaskSetTabularConfig_6p(TaskSetTabularConfig):
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    l1: float
    l2: float


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class TaskSetTabularConfig_8p(TaskSetTabularConfig):
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    l1: float
    l2: float
    linear_decay: float
    exponential_decay: float


C = TypeVar("C", bound=TaskSetTabularConfig)


@dataclass(frozen=True)  # type: ignore[misc]
class TaskSetTabularResult(Result[C, int]):
    metric_defs: ClassVar[Mapping[str, Metric]] = {
        "train_loss": Metric(minimize=True, bounds=(0, np.inf)),
        # Not sure why they have 2 valid losses...
        "valid1_loss": Metric(minimize=True, bounds=(0, np.inf)),
        "valid2_loss": Metric(minimize=True, bounds=(0, np.inf)),
        "test_loss": Metric(minimize=True, bounds=(0, np.inf)),
        "cost": Metric(minimize=True, bounds=(0, np.inf)),
    }
    default_value_metric: ClassVar[str] = "valid1_loss"
    default_value_metric_test: ClassVar[str] = "test_loss"
    default_cost_metric: ClassVar[str] = "cost"

    train_loss: Metric.Value
    valid_loss: Metric.Value
    valid2_loss: Metric.Value
    test_loss: Metric.Value
    train_cost: Metric.Value


class TaskSetTabularBenchmark(
    TabularBenchmark[TaskSetTabularConfig, TaskSetTabularResult, int],
):
    """The taskset tabular benchmark.

    NOTE: The "epoch" that is used as the index is not actually the epoch
    but is used to keep it inline with other benchmarks. Please refer
    to the "step" column for the measurement of fidelity used. We can do
    so pretty safely for benchmarking as the steps are incremented in a uniform
    range.
    """

    task_ids: ClassVar[tuple[str, ...]] = (
        "Associative_GRU128_BS128_Pairs10_Tokens50",
        "Associative_GRU256_BS128_Pairs20_Tokens50",
        "Associative_LSTM128_BS128_Pairs10_Tokens50",
        "Associative_LSTM128_BS128_Pairs20_Tokens50",
        "Associative_LSTM128_BS128_Pairs5_Tokens20",
        "Associative_LSTM256_BS128_Pairs20_Tokens50",
        "Associative_LSTM256_BS128_Pairs40_Tokens100",
        "Associative_VRNN128_BS128_Pairs10_Tokens50",
        "Associative_VRNN256_BS128_Pairs20_Tokens50",
        "char_rnn_language_model_family",
        "conv_fc_family",
        "conv_pooling_family",
        "Copy_GRU128_BS128_Length20_Tokens10",
        "Copy_GRU256_BS128_Length40_Tokens50",
        "Copy_LSTM128_BS128_Length20_Tokens10",
        "Copy_LSTM128_BS128_Length20_Tokens20",
        "Copy_LSTM128_BS128_Length50_Tokens5",
        "Copy_LSTM128_BS128_Length5_Tokens10",
        "Copy_LSTM256_BS128_Length40_Tokens50",
        "Copy_VRNN128_BS128_Length20_Tokens10",
        "Copy_VRNN256_BS128_Length40_Tokens50",
        "FixedImageConvAE_cifar10_32x32x32x32x32_bs128",
        "FixedImageConvAE_cifar10_32x64x8x64x32_bs128",
        "FixedImageConvAE_mnist_32x32x32x32x32_bs128",
        "FixedImageConvAE_mnist_32x64x32x64x32_bs512",
        "FixedImageConvAE_mnist_32x64x8x64x32_bs128",
        "FixedImageConv_cifar100_32x64x128_FC64x32_tanh_variance_scaling_bs64",
        "FixedImageConv_cifar100_32x64x64_flatten_bs128",
        "FixedImageConv_cifar100_bn_32x64x128x128_bs128",
        "FixedImageConv_cifar10_32x64x128_flatten_FC64x32_tanh_he_bs8",
        "FixedImageConv_cifar10_32x64x128_flatten_FC64x32_tanh_variance_scaling_bs64",
        "FixedImageConv_cifar10_32x64x128_he_bs64",
        "FixedImageConv_cifar10_32x64x128_largenormal_bs64",
        "FixedImageConv_cifar10_32x64x128_normal_bs64",
        "FixedImageConv_cifar10_32x64x128_smallnormal_bs64",
        "FixedImageConv_cifar10_32x64x128x128x128_avg_he_bs64",
        "FixedImageConv_cifar10_32x64x64_bs128",
        "FixedImageConv_cifar10_32x64x64_fc_64_bs128",
        "FixedImageConv_cifar10_32x64x64_flatten_bs128",
        "FixedImageConv_cifar10_32x64x64_tanh_bs64",
        "FixedImageConv_cifar10_batchnorm_32x32x32x64x64_bs128",
        "FixedImageConv_cifar10_batchnorm_32x64x64_bs128",
        "FixedImageConv_coil10032x32_bn_32x64x128x128_bs128",
        "FixedImageConv_colorectalhistology32x32_32x64x64_flatten_bs128",
        "FixedImageConv_food10164x64_Conv_32x64x64_flatten_bs64",
        "FixedImageConv_food101_batchnorm_32x32x32x64x64_bs128",
        "FixedImageConv_mnist_32x64x64_fc_64_bs128",
        "FixedImageConv_sun39732x32_bn_32x64x128x128_bs128",
        "FixedImageConvVAE_cifar10_32x64x128x64x128x64x32_bs128",
        "FixedImageConvVAE_cifar10_32x64x128x64x128x64x32_bs512",
        "FixedImageConvVAE_cifar10_32x64x128x64x32_bs128",
        "FixedImageConvVAE_cifar10_64x128x256x128x256x128x64_bs128",
        "FixedImageConvVAE_mnist_32x32x32x32x32_bs128",
        "FixedImageConvVAE_mnist_32x64x32x64x32_bs128",
        "FixedImageConvVAE_mnist_64x128x128x128x64_bs128",
        "FixedLM_lm1b_patch128_GRU128_embed64_avg_bs128",
        "FixedLM_lm1b_patch128_GRU256_embed64_avg_bs128",
        "FixedLM_lm1b_patch128_GRU64_embed64_avg_bs128",
        "FixedLM_lm1b_patch128_LSTM128_embed64_avg_bs128",
        "FixedLM_lm1b_patch128_LSTM256_embed64_avg_bs128",
        "FixedMAF_cifar10_3layer_bs64",
        "FixedMAF_mnist_2layer_bs64",
        "FixedMAF_mnist_3layer_thin_bs64",
        "FixedMLPAE_cifar10_128x32x128_bs128",
        "FixedMLPAE_mnist_128x32x128_bs128",
        "FixedMLPAE_mnist_32x32x32_bs128",
        "FixedMLP_cifar10_BatchNorm_128x128x128_relu_bs128",
        "FixedMLP_cifar10_BatchNorm_64x64x64x64x64_relu_bs128",
        "FixedMLP_cifar10_ce_128x128x128_relu_bs128",
        "FixedMLP_cifar10_Dropout02_128x128_relu_bs128",
        "FixedMLP_cifar10_Dropout05_128x128_relu_bs128",
        "FixedMLP_cifar10_Dropout08_128x128_relu_bs128",
        "FixedMLP_cifar10_LayerNorm_128x128x128_relu_bs128",
        "FixedMLP_cifar10_LayerNorm_128x128x128_tanh_bs128",
        "FixedMLP_cifar10_mse_128x128x128_relu_bs128",
        "FixedMLP_food10132x32_ce_128x128x128_relu_bs128",
        "FixedMLP_food10132x32_mse_128x128x128_relu_bs128",
        "FixedMLP_mnist_ce_128x128x128_relu_bs128",
        "FixedMLP_mnist_mse_128x128x128_relu_bs128",
        "FixedMLPVAE_cifar101_128x128x32x128x128_bs128",
        "FixedMLPVAE_cifar101_128x32x128_bs128",
        "FixedMLPVAE_food10132x32_128x64x32x64x128_bs64",
        "FixedMLPVAE_mnist_128x128x8x128_bs128",
        "FixedMLPVAE_mnist_128x64x32x64x128_bs64",
        "FixedMLPVAE_mnist_128x8x128x128_bs128",
        "FixedNVP_mnist_2layer_bs64",
        "FixedNVP_mnist_3layer_thin_bs64",
        "FixedNVP_mnist_5layer_bs64",
        "FixedNVP_mnist_5layer_thin_bs64",
        "FixedNVP_mnist_9layer_thin_bs16",
        "FixedTextRNNClassification_imdb_patch128_LSTM128_avg_bs64",
        "FixedTextRNNClassification_imdb_patch128_LSTM128_bs64",
        "FixedTextRNNClassification_imdb_patch128_LSTM128_embed128_bs64",
        "FixedTextRNNClassification_imdb_patch32_GRU128_bs128",
        "FixedTextRNNClassification_imdb_patch32_GRU64_avg_bs128",
        "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_avg_bs128",
        "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_last_bs128",
        "FixedTextRNNClassification_imdb_patch32_LSTM128_bs128",
        "FixedTextRNNClassification_imdb_patch32_LSTM128_E128_bs128",
        "FixedTextRNNClassification_imdb_patch32_VRNN128_tanh_bs128",
        "FixedTextRNNClassification_imdb_patch32_VRNN64_relu_avg_bs128",
        "FixedTextRNNClassification_imdb_patch32_VRNN64_tanh_avg_bs128",
        "Imagenet32x30_FC_VAE_128x64x32x64x128_relu_bs256",
        "losg_tasks_family",
        "maf_family",
        "mlp_ae_family",
        "mlp_family",
        "mlp_vae_family",
        "Mnist_Conv_32x16x64_flatten_FC32_tanh_bs32",
        "nvp_family",
        "quadratic_family",
        "rnn_text_classification_family",
        "TwoD_Ackley",
        "TwoD_Beale",
        "TwoD_Bowl1",
        "TwoD_Bowl10",
        "TwoD_Bowl100",
        "TwoD_Bowl1000",
        "TwoD_Rosenbrock",
        "TwoD_StyblinskiTang",
        "word_rnn_language_model_family",
    )
    optimizers: ClassVar[tuple[str, ...]] = (
        "adam1p_wide_grid_1k",
        "adam4p_wide_grid_1k",
        "adam6p_wide_grid_1k",
        "adam8p_wide_grid_1k",
        "nadamw_grid_1k",
    )
    illegal_combinations: ClassVar[set[tuple[str, str]]] = {
        ("char_rnn_language_model_family", "adam1p_wide_grid_1k"),
        ("char_rnn_language_model_family", "adam4p_wide_grid_1k"),
        ("char_rnn_language_model_family", "adam6p_wide_grid_1k"),
        ("char_rnn_language_model_family", "adam8p_wide_grid_1k"),
        ("char_rnn_language_model_family", "nadamw_grid_1k"),
        ("conv_fc_family", "adam1p_wide_grid_1k"),
        ("conv_fc_family", "adam4p_wide_grid_1k"),
        ("conv_fc_family", "adam6p_wide_grid_1k"),
        ("conv_fc_family", "adam8p_wide_grid_1k"),
        ("conv_fc_family", "nadamw_grid_1k"),
        ("conv_pooling_family", "adam1p_wide_grid_1k"),
        ("conv_pooling_family", "adam4p_wide_grid_1k"),
        ("conv_pooling_family", "adam6p_wide_grid_1k"),
        ("conv_pooling_family", "adam8p_wide_grid_1k"),
        ("conv_pooling_family", "nadamw_grid_1k"),
        ("losg_tasks_family", "adam1p_wide_grid_1k"),
        ("losg_tasks_family", "adam4p_wide_grid_1k"),
        ("losg_tasks_family", "adam6p_wide_grid_1k"),
        ("losg_tasks_family", "adam8p_wide_grid_1k"),
        ("losg_tasks_family", "nadamw_grid_1k"),
        ("maf_family", "adam1p_wide_grid_1k"),
        ("maf_family", "adam4p_wide_grid_1k"),
        ("maf_family", "adam6p_wide_grid_1k"),
        ("maf_family", "adam8p_wide_grid_1k"),
        ("maf_family", "nadamw_grid_1k"),
        ("mlp_ae_family", "adam1p_wide_grid_1k"),
        ("mlp_ae_family", "adam4p_wide_grid_1k"),
        ("mlp_ae_family", "adam6p_wide_grid_1k"),
        ("mlp_ae_family", "adam8p_wide_grid_1k"),
        ("mlp_ae_family", "nadamw_grid_1k"),
        ("mlp_family", "adam1p_wide_grid_1k"),
        ("mlp_family", "adam4p_wide_grid_1k"),
        ("mlp_family", "adam6p_wide_grid_1k"),
        ("mlp_family", "adam8p_wide_grid_1k"),
        ("mlp_family", "nadamw_grid_1k"),
        ("mlp_vae_family", "adam1p_wide_grid_1k"),
        ("mlp_vae_family", "adam4p_wide_grid_1k"),
        ("mlp_vae_family", "adam6p_wide_grid_1k"),
        ("mlp_vae_family", "adam8p_wide_grid_1k"),
        ("mlp_vae_family", "nadamw_grid_1k"),
        ("nvp_family", "adam1p_wide_grid_1k"),
        ("nvp_family", "adam4p_wide_grid_1k"),
        ("nvp_family", "adam6p_wide_grid_1k"),
        ("nvp_family", "adam8p_wide_grid_1k"),
        ("nvp_family", "nadamw_grid_1k"),
        ("quadratic_family", "adam1p_wide_grid_1k"),
        ("quadratic_family", "adam4p_wide_grid_1k"),
        ("quadratic_family", "adam6p_wide_grid_1k"),
        ("quadratic_family", "adam8p_wide_grid_1k"),
        ("quadratic_family", "nadamw_grid_1k"),
        ("rnn_text_classification_family", "adam1p_wide_grid_1k"),
        ("rnn_text_classification_family", "adam4p_wide_grid_1k"),
        ("rnn_text_classification_family", "adam6p_wide_grid_1k"),
        ("rnn_text_classification_family", "adam8p_wide_grid_1k"),
        ("rnn_text_classification_family", "nadamw_grid_1k"),
        ("word_rnn_language_model_family", "adam1p_wide_grid_1k"),
        ("word_rnn_language_model_family", "adam4p_wide_grid_1k"),
        ("word_rnn_language_model_family", "adam6p_wide_grid_1k"),
        ("word_rnn_language_model_family", "adam8p_wide_grid_1k"),
        ("word_rnn_language_model_family", "nadamw_grid_1k"),
    }

    def __init__(
        self,
        task_id: str,
        optimizer: str,
        datadir: str | Path | None = None,
        *,
        seed: int | None = None,
        prior: str | Path | TaskSetTabularConfig | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
        value_metric: str | None = None,
        value_metric_test: str | None = None,
        cost_metric: str | None = None,
    ) -> None:
        """Initialize a taskset tabular benchmark.

        Args:
            task_id: The task id to use.
            optimizer: The optimizer to use.
            datadir: The directory to use for the data.
            remove_constants: Whether or not to remove constants from the
                config space.
            seed: The seed to use for the benchmark.
            prior: The prior to use for the benchmark.
            perturb_prior: The perturbation to use for the prior.
            value_metric: The value metric to use for the benchmark.
            value_metric_test: The test value metric to use for the benchmark.
            cost_metric: The cost metric to use for the benchmark.
        """
        cls = self.__class__
        if task_id not in cls.task_ids:
            raise ValueError(
                f"Unknown task {task_id}, must be one of {cls.task_ids}",
            )
        if optimizer not in cls.optimizers:
            raise ValueError(
                f"Unknown task {optimizer}, must be one of {cls.optimizers}",
            )

        if (task_id, optimizer) in cls.illegal_combinations:
            raise ValueError(
                f"These are the illegal combinations: {cls.illegal_combinations}.",
                f"\nCannot use task {task_id} with optimizer {optimizer}.",
            )

        if datadir is None:
            from mfpbench.setup_benchmark import TaskSetabularSource

            datadir = TaskSetabularSource.default_location()

        filename = f"{task_id}-{optimizer}_10000_replica5.parquet"
        table_path = Path(datadir) / filename
        if not table_path.exists():
            raise FileNotFoundError(
                f"Could not find table {table_path}."
                f"`python -m mfpbench download --status --data-dir {datadir}",
            )

        # Reading table
        table = pd.read_parquet(table_path)

        space = _get_raw_taskset_space(
            name=task_id,
            seed=seed,
            optimizer=optimizer,
        )

        if "1p" in optimizer:
            config_type = TaskSetTabularConfig_1p
        elif "4p" in optimizer:
            config_type = TaskSetTabularConfig_4p
        elif "6p" in optimizer:
            config_type = TaskSetTabularConfig_6p
        elif "8p" in optimizer:
            config_type = TaskSetTabularConfig_8p
        else:
            raise ValueError("Cannot recognize optimizer!")

        super().__init__(
            table=table,  # type: ignore
            name=task_id,
            id_key="config_id",
            fidelity_key="epoch",
            result_type=TaskSetTabularResult,
            config_type=config_type,  # type: ignore
            info_keys=[],
            value_metric=value_metric,
            value_metric_test=value_metric_test,
            cost_metric=cost_metric,
            space=space,
            seed=seed,
            prior=prior,
            perturb_prior=perturb_prior,
        )
