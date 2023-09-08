from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import override

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from mfpbench.pd1.benchmark import PD1Benchmark, PD1Config, PD1ResultTransformer


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class PD1Config_lm1b_transformer_2048(PD1Config):
    @override
    def validate(self) -> None:
        assert 0.010543 <= self.lr_decay_factor <= 9.885653e-01
        assert 0.000010 <= self.lr_initial <= 9.986256e00
        assert 0.100811 <= self.lr_power <= 1.999659e00
        assert 0.000059 <= self.opt_momentum <= 9.989986e-01


class PD1lm1b_transformer_2048(PD1Benchmark):
    fidelity_name = "epoch"
    fidelity_range = (1, 74, 1)

    Config = PD1Config_lm1b_transformer_2048
    Result = PD1ResultTransformer

    pd1_dataset = "lm1b"
    pd1_model = "transformer"
    pd1_batchsize = 2048
    pd1_metrics = ("valid_error_rate", "train_cost")

    @classmethod
    @override
    def _create_space(cls, seed: int | None = None) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "lr_decay_factor",
                    lower=0.010543,
                    upper=9.885653e-01,
                ),
                UniformFloatHyperparameter(
                    "lr_initial",
                    lower=0.000010,
                    upper=9.986256e00,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "lr_power",
                    lower=0.100811,
                    upper=1.999659e00,
                ),
                UniformFloatHyperparameter(
                    "opt_momentum",
                    lower=0.000059,
                    upper=9.989986e-01,
                    log=True,
                ),
            ],
        )
        return cs
