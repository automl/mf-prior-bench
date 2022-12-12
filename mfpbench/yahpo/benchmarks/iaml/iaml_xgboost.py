from __future__ import annotations

from dataclasses import dataclass
from typing import no_type_check

from typing_extensions import Literal

from mfpbench.yahpo.benchmarks.iaml.iaml import IAMLBenchmark, IAMLConfig, IAMLResult


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class IAMLxgboostConfig(IAMLConfig):

    nrounds: int  # log
    subsample: float
    alpha: float  # log
    _lambda: float  # log
    booster: Literal["gblinear", "gbtree", "dart"]

    colsample_bylevel: float | None = None
    colsample_bytree: float | None = None
    eta: float | None = None  # log
    gamma: float | None = None  # log
    max_depth: int | None = None
    min_child_weight: float | None = None  # log
    rate_drop: float | None = None
    skip_drop: float | None = None

    @no_type_check
    def validate(self) -> None:
        """Validate this config."""
        assert self.booster in ["gblinear", "gbtree", "dart"]
        assert 0.00010000000000000009 <= self.alpha <= 999.9999999999998
        assert 0.00010000000000000009 <= self._lambda <= 999.9999999999998
        assert 7 <= self.nrounds <= 2981
        assert 0.1 <= self.subsample <= 1.0

        if self.colsample_bylevel is not None:
            assert self.booster in ["dart", "gbtree"]
            assert 0.01 <= self.colsample_bylevel <= 1.0

        if self.colsample_bytree is not None:
            assert self.booster in ["dart", "gbtree"]
            assert 0.01 <= self.colsample_bytree <= 1.0

        if self.eta is not None:
            assert self.booster in ["dart", "gbtree"]
            assert 0.00010000000000000009 <= self.eta <= 1.0

        if self.gamma is not None:
            assert self.booster in ["dart", "gbtree"]
            assert 0.00010000000000000009 <= self.gamma <= 6.999999999999999

        if self.max_depth is not None:
            assert self.booster in ["dart", "gbtree"]
            assert 1 <= self.max_depth <= 15

        if self.min_child_weight is not None:
            assert self.booster in ["dart", "gbtree"]
            assert 2.718281828459045 <= self.min_child_weight <= 149.99999999999997

        if self.rate_drop is not None:
            assert self.booster in ["dart"]
            assert 0.0 <= self.rate_drop <= 1.0

        if self.skip_drop is not None:
            assert self.booster in ["dart"]
            assert 0.0 <= self.skip_drop <= 1.0


@dataclass(frozen=True)
class IAMLxgboostResult(IAMLResult):
    config: IAMLxgboostConfig


class IAMLxgboostBenchmark(IAMLBenchmark):
    name = "iaml_xgboost"
    Result = IAMLxgboostResult
    Config = IAMLxgboostConfig
    has_conditionals = True

    _replacements_hps = [("_lambda", "lambda")]

    instances = ["40981", "41146", "1489", "1067"]
