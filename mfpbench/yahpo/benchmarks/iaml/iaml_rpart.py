from __future__ import annotations

from dataclasses import dataclass
from typing import no_type_check

from mfpbench.yahpo.benchmarks.iaml.iaml import IAMLBenchmark, IAMLConfig, IAMLResult


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class IAMLrpartConfig(IAMLConfig):

    cp: float  # log
    maxdepth: int
    minbucket: int
    minsplit: int

    @no_type_check
    def validate(self) -> None:
        """Validate this config."""
        assert 0.00010000000000000009 <= self.cp <= 1.0
        assert 1 <= self.maxdepth <= 30
        assert 1 <= self.minbucket <= 100
        assert 1 <= self.minsplit <= 100


@dataclass(frozen=True)
class IAMLrpartResult(IAMLResult):
    config: IAMLrpartConfig


class IAMLrpartBenchmark(IAMLBenchmark):
    name = "iaml_rpart"
    Result = IAMLrpartResult
    Config = IAMLrpartConfig
    has_conditionals = False

    instances = ["40981", "41146", "1489", "1067"]
