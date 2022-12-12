from __future__ import annotations

from dataclasses import dataclass
from typing import no_type_check

from typing_extensions import Literal

from mfpbench.yahpo.benchmarks.iaml.iaml import IAMLBenchmark, IAMLConfig, IAMLResult


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class IAMLSuperConfig(IAMLConfig):
    """Config has conditionals and as such, we use None to indicate not set."""

    learner_id: Literal["glmnet", "ranger", "rpart", "xgboost"]

    glmnet__alpha: float | None = None
    glmnet__s: float | None = None  # log

    ranger__min__node__size: int | None = None
    ranger__mtry__power: int | None = None
    ranger__num__random__splits: int | None = None
    ranger__num__trees: int | None = None
    ranger__respect__unordered__factors: Literal[
        "ignore", "order", "partition"
    ] | None = None
    ranger__sample__fraction: float | None = None
    ranger__splitrule: Literal["gini", "extratrees"] | None = None

    rpart__cp: float | None = None  # log
    rpart__maxdepth: int | None = None
    rpart__minbucket: int | None = None
    rpart__minsplit: int | None = None

    xgboost__alpha: float | None = None  # log
    xgboost__booster: Literal["gblinear", "gbtree", "dart"] | None = None
    xgboost__colsample_bylevel: float | None = None
    xgboost__colsample_bytree: float | None = None
    xgboost__eta: float | None = None  # log

    xgboost__gamma: float | None = None  # log
    xgboost__lambda: float | None = None  # log
    xgboost__max_depth: int | None = None
    xgboost__min_child_weight: float | None = None  # log
    xgboost__nrounds: int | None = None  # log
    xgboost__rate_drop: float | None = None
    xgboost__skip_drop: float | None = None
    xgboost__subsample: float | None = None

    @no_type_check
    def validate(self) -> None:
        """Validate this config."""
        assert self.learner_id in ["glmnet", "ranger", "rpart", "xgboost"]

        # We do some conditional checking here
        learner = self.learner_id

        # We filter out all attributes except for those that must always be contained
        # or are the selected learner, ...
        attrs = [
            attr
            for attr in dir(self)
            if not attr.startswith("__")
            or not attr.startswith(learner)
            or attr in ["learner_id"]
        ]

        # ... the remaining must always have None set then
        for attr in attrs:
            assert attr is None

        if learner == "glmnet":
            assert 0.0 <= self.glmnet__alpha <= 1.0
            assert 0.00010000000000000009 <= self.glmnet__s <= 999.9999999999998

        elif learner == "rpart":
            assert 0.00010000000000000009 <= self.rpart__cp <= 1.0
            assert 1 <= self.rpart__maxdepth <= 30
            assert 1 <= self.rpart__minbucket <= 100
            assert 1 <= self.rpart__minsplit <= 100

        elif learner == "ranger":
            assert 1 <= self.ranger__min__node__size <= 100
            assert 0 <= self.ranger__mtry__power <= 1
            assert 1 <= self.ranger__num__trees <= 2000
            assert self.ranger__respect__unordered__factors in [
                "ignore",
                "order",
                "partition",
            ]
            assert 0.1 <= self.ranger__sample__fraction <= 1.0
            assert self.ranger__splitrule in ["gini", "extratrees"]

            if self.ranger__num__random__splits is not None:
                assert self.ranger__splitrule == "extratrees"
                assert 1 <= self.ranger__num__random__splits <= 100

        elif learner == "xgboost":
            assert self.xgboost__booster in ["gblinear", "gbtree", "dart"]
            assert 0.00010000000000000009 <= self.xgboost__alpha <= 999.9999999999998
            assert 0.00010000000000000009 <= self.xgboost__lambda <= 999.9999999999998
            assert 7 <= self.xgboost__nrounds <= 2981
            assert 0.1 <= self.xgboost__subsample <= 1.0

            if self.xgboost__colsample_bylevel is not None:
                assert self.xgboost__booster in ["dart", "gbtree"]
                assert 0.01 <= self.xgboost__colsample_bylevel <= 1.0

            if self.xgboost__colsample_bytree is not None:
                assert self.xgboost__booster in ["dart", "gbtree"]
                assert 0.01 <= self.xgboost__colsample_bytree <= 1.0

            if self.xgboost__eta is not None:
                assert self.xgboost__booster in ["dart", "gbtree"]
                assert 0.00010000000000000009 <= self.eta <= 1.0

            if self.xgboost__gamma is not None:
                assert self.xgboost__booster in ["dart", "gbtree"]
                assert 0.00010000000000000009 <= self.gamma <= 6.999999999999999

            if self.xgboost__max_depth is not None:
                assert self.xgboost__booster in ["dart", "gbtree"]
                assert 1 <= self.xgboost__max_depth <= 15

            if self.xgboost__min_child_weight is not None:
                assert self.xgboost__booster in ["dart", "gbtree"]
                assert (
                    2.718281828459045
                    <= self.xgboost__min_child_weight
                    <= 149.99999999999997
                )

            if self.rate_drop is not None:
                assert self.xgboost__booster in ["dart"]
                assert 0.0 <= self.xgboost__rate_drop <= 1.0

            if self.xgboost__skip_drop is not None:
                assert self.xgboost__booster in ["dart"]
                assert 0.0 <= self.xgboost__skip_drop <= 1.0

        else:
            raise NotImplementedError()


@dataclass(frozen=True)
class IAMLSuperResult(IAMLResult):
    config: IAMLSuperConfig


class IAMLSuperBenchmark(IAMLBenchmark):
    name = "iaml_super"

    Result = IAMLSuperResult
    Config = IAMLSuperConfig

    has_conditionals = True

    instances = ["40981", "41146", "1489", "1067"]
