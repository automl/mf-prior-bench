from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import override

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from mfpbench.pd1.benchmark import PD1Benchmark, PD1Config, PD1ResultTransformer


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class PD1Config_imagenet_resnet_512(PD1Config):
    @override
    def validate(self) -> None:
        assert 0.010294 <= self.lr_decay_factor <= 0.989753
        assert 0.000010 <= self.lr_initial <= 9.774312
        assert 0.100225 <= self.lr_power <= 1.999326
        assert 0.000059 <= self.opt_momentum <= 0.998993


class PD1imagenet_resnet_512(PD1Benchmark):
    fidelity_name = "epoch"
    fidelity_range = (3, 99, 1)
    Config = PD1Config_imagenet_resnet_512
    Result = PD1ResultTransformer

    pd1_dataset = "imagenet"
    pd1_model = "resnet"
    pd1_batchsize = 512
    pd1_metrics = ("valid_error_rate", "train_cost")

    @classmethod
    def _create_space(cls, seed: int | None = None) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "lr_decay_factor",
                    lower=0.010294,
                    upper=0.989753,
                ),
                UniformFloatHyperparameter(
                    "lr_initial",
                    lower=0.000010,
                    upper=9.774312,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "lr_power",
                    lower=0.100225,
                    upper=1.999326,
                ),
                UniformFloatHyperparameter(
                    "opt_momentum",
                    lower=0.000059,
                    upper=0.998993,
                    log=True,
                ),
            ],
        )
        return cs
