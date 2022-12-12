from __future__ import annotations

from dataclasses import dataclass
from typing import no_type_check

from mfpbench.yahpo.benchmarks.iaml.iaml import IAMLBenchmark, IAMLConfig, IAMLResult


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class IAMLglmnetConfig(IAMLConfig):

    alpha: float
    s: float  # log

    @no_type_check
    def validate(self) -> None:
        """Validate this config."""
        assert 0.0 <= self.alpha <= 1.0
        assert 0.00010000000000000009 <= self.s <= 999.9999999999998


@dataclass(frozen=True)
class IAMLglmnetResult(IAMLResult):
    config: IAMLglmnetConfig


class IAMLglmnetBenchmark(IAMLBenchmark):
    name = "iaml_glmnet"
    Result = IAMLglmnetResult
    Config = IAMLglmnetConfig
    has_conditionals = False

    instances = ["40981", "41146", "1489", "1067"]
