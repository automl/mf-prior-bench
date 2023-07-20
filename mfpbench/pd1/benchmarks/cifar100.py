from __future__ import annotations

from dataclasses import dataclass

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from mfpbench.pd1.benchmark import PD1Benchmark
from mfpbench.pd1.config import PD1Config
from mfpbench.pd1.result import PD1ResultTransformer


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class PD1Config_cifar100_wideresnet_2048(PD1Config):
    def validate(self) -> None:
        assert 0.010093 <= self.lr_decay_factor <= 0.989012
        assert 0.000010 <= self.lr_initial <= 9.779176
        assert 0.100708 <= self.lr_power <= 1.999376
        assert 0.000059 <= self.opt_momentum <= 0.998993


class PD1cifar100_wideresnet_2048(PD1Benchmark):

    fidelity_name = "epoch"

    Config = PD1Config_cifar100_wideresnet_2048
    Result = PD1ResultTransformer

    dataset = "cifar100"
    model = "wide_resnet"
    batchsize = 2048
    metrics = ("valid_error_rate", "train_cost")

    fidelity_range = (45, 199, 1)

    @classmethod
    def _create_space(cls, seed: int | None = None) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "lr_decay_factor",
                    lower=0.010093,
                    upper=0.989012,
                ),
                UniformFloatHyperparameter(
                    "lr_initial",
                    lower=0.000010,
                    upper=9.779176,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "lr_power",
                    lower=0.100708,
                    upper=1.999376,
                ),
                UniformFloatHyperparameter(
                    "opt_momentum",
                    lower=0.000059,
                    upper=0.998993,
                    log=True,
                ),
            ]
        )
        return cs
