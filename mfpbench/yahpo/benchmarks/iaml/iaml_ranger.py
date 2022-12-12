from __future__ import annotations

from dataclasses import dataclass
from typing import no_type_check

from typing_extensions import Literal

from mfpbench.yahpo.benchmarks.iaml.iaml import IAMLBenchmark, IAMLConfig, IAMLResult


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class IAMLrangerConfig(IAMLConfig):

    min__node__size: int
    mtry__power: int
    num__trees: int
    respect__unordered__factors: Literal["ignore", "order", "partition"]
    sample__fraction: float
    splitrule: Literal["gini", "extratrees"]

    num__random__splits: int | None = None

    @no_type_check
    def validate(self) -> None:
        """Validate this config."""
        assert 1 <= self.min__node__size <= 100
        assert 0 <= self.mtry__power <= 1
        assert 1 <= self.num__trees <= 2000
        assert self.respect__unordered__factors in [
            "ignore",
            "order",
            "partition",
        ]
        assert 0.1 <= self.sample__fraction <= 1.0
        assert self.splitrule in ["gini", "extratrees"]

        if self.num__random__splits is not None:
            assert self.splitrule == "extratrees"
            assert 1 <= self.num__random__splits <= 100


@dataclass(frozen=True)
class IAMLrangerResult(IAMLResult):
    config: IAMLrangerConfig


class IAMLrangerBenchmark(IAMLBenchmark):
    name = "iaml_ranger"
    Result = IAMLrangerResult
    Config = IAMLrangerConfig
    has_conditionals = True

    instances = ["40981", "41146", "1489", "1067"]
