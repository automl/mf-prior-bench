from __future__ import annotations

from dataclasses import dataclass

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from mfpbench.pd1.benchmark import PD1Benchmark
from mfpbench.pd1.config import PD1Config
from mfpbench.pd1.result import PD1ResultTransformer


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class PD1Config_lm1b_transformer_2048(PD1Config):
    def validate(self) -> None:
        assert 0.010543 <= self.lr_decay_factor <= 9.885653e-01
        assert 0.000010 <= self.lr_initial <= 9.986256e00
        assert 0.100811 <= self.lr_power <= 1.999659e00
        assert 0.000059 <= self.opt_momentum <= 9.989986e-01


class PD1lm1b_transformer_2048(PD1Benchmark):

    fidelity_name = "epoch"

    Config = PD1Config_lm1b_transformer_2048
    Result = PD1ResultTransformer

    dataset = "lm1b"
    model = "transformer"
    batchsize = 2048
    metrics = ("valid_error_rate", "train_cost")

    fidelity_range = (1, 74, 1)

    @classmethod
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
            ]
        )
        return cs
