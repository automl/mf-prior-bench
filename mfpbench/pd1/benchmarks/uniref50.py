from __future__ import annotations

from dataclasses import dataclass

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from mfpbench.pd1.benchmark import PD1Benchmark
from mfpbench.pd1.config import PD1Config
from mfpbench.pd1.result import PD1ResultTransformer


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class PD1Config_uniref50_transformer_128(PD1Config):
    def validate(self) -> None:
        assert 0.0111588123 <= self.lr_decay_factor <= 0.9898713967
        assert 1.00564e-05 <= self.lr_initial <= 0.4429248972
        assert 0.1001570089 <= self.lr_power <= 1.9989163336
        assert 5.86114e-05 <= self.opt_momentum <= 0.9989940217


class PD1uniref50_transformer_128(PD1Benchmark):

    fidelity_name = "epoch"

    Config = PD1Config_uniref50_transformer_128
    Result = PD1ResultTransformer

    dataset = "uniref50"
    model = "transformer"
    batchsize = 128
    metrics = ("valid_error_rate", "train_cost")

    fidelity_range = (1, 22, 1)

    @classmethod
    def _create_space(cls, seed: int | None = None) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "lr_decay_factor",
                    lower=0.0111588123,
                    upper=0.9898713967,
                ),
                UniformFloatHyperparameter(
                    "lr_initial",
                    lower=1.00564e-05,
                    upper=0.4429248972,
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "lr_power",
                    lower=0.1001570089,
                    upper=1.9989163336,
                ),
                UniformFloatHyperparameter(
                    "opt_momentum",
                    lower=5.86114e-05,
                    upper=0.9989940217,
                    log=True,
                ),
            ]
        )
        return cs
