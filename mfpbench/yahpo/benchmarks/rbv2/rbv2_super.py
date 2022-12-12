from __future__ import annotations

from dataclasses import dataclass
from typing import no_type_check

from typing_extensions import Literal

from mfpbench.yahpo.benchmarks.rbv2.rbv2 import RBV2Benchmark, RBV2Config, RBV2Result


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class RBV2SuperConfig(RBV2Config):
    """Config has conditionals and as such, we use None to indicate not set."""

    learner_id: Literal["aknn", "glmnet", "ranger", "rpart", "svm", "xgboost"]
    num__impute__selected__cpo: Literal["impute.mean", "impute.median", "impute.hist"]

    # aknn
    aknn__M: int | None = None  # (18, 50)
    aknn__distance: Literal["l2", "cosine", "ip"] | None = None
    aknn__ef: int | None = None  # (7, 403),  log
    aknn__ef_construction: int | None = None  # (7, 1097),  log
    aknn__k: int | None = None  # (1, 50)

    glmnet__alpha: float | None = None  # (0.0, 1.0)
    glmnet__s: float | None = None  # (0.0009118819655545162, 1096.6331584284585), log

    ranger__min__node__size: int | None = None  # (1, 100)
    ranger__mtry__power: int | None = None  # (0, 1)
    ranger__num__random__splits: int | None = None  # (1, 100)
    ranger__num__trees: int | None = None  # (1, 2000)
    ranger__respect__unordered__factors: Literal[
        "ignore", "order", "partition"
    ] | None = None
    ranger__sample__fraction: float | None = None  # (0.1, 1.0)
    ranger__splitrule: Literal["gini", "extratrees"] | None = None

    rpart__cp: float | None = None  # (0.0009118819655545162, 1.0), log
    rpart__maxdepth: int | None = None  # (1, 30)
    rpart__minbucket: int | None = None  # (1, 100)
    rpart__minsplit: int | None = None  # (1, 100)

    svm__cost: float | None = None  # (4.5399929762484854e-05, 22026.465794806718), log
    svm__degree: int | None = None  # (2, 5)
    svm__gamma: float | None = None  # (4.5399929762484854e-05, 22026.465794806718), log
    svm__kernel: Literal["linear", "polynomial", "radial"] | None = None
    svm__tolerance: float | None = None  # (4.5399929762484854e-05, 2.0) log

    # (0.0009118819655545162, 1096.6331584284585), log
    xgboost__alpha: float | None = None
    xgboost__booster: Literal["gblinear", "gbtree", "dart"] | None = None
    xgboost__colsample_bylevel: float | None = None  # (0.01, 1.0)
    xgboost__colsample_bytree: float | None = None  # (0.01, 1.0)
    xgboost__eta: float | None = None  # (0.0009118819655545162, 1.0)  log
    # (4.5399929762484854e-05, 7.38905609893065), log
    xgboost__gamma: float | None = None
    # (0.0009118819655545162, 1096.6331584284585), log
    xgboost__lambda: float | None = None
    xgboost__max_depth: int | None = None  # (1, 15)
    # (2.718281828459045, 148.4131591025766),  log
    xgboost__min_child_weight: float | None = None
    xgboost__nrounds: int | None = None  # (7, 2981), log
    xgboost__rate_drop: float | None = None  # (0.0, 1.0)
    xgboost__skip_drop: float | None = None  # (0.0, 1.0)
    xgboost__subsample: float | None = None  # (0.1, 1.0)

    @no_type_check
    def validate(self) -> None:
        """Validate this config."""
        assert self.learner_id in [
            "aknn",
            "glmnet",
            "ranger",
            "rpart",
            "svm",
            "xgboost",
        ]

        assert self.num__impute__selected__cpo in [
            "impute.mean",
            "impute.median",
            "impute.hist",
        ]

        # We do some conditional checking here
        learner = self.learner_id

        # We filter out all attributes except for those that must always be contained
        # or are the selected learner, ...
        attrs = [
            attr
            for attr in dir(self)
            if not attr.startswith("__")
            or not attr.startswith(learner)
            or attr in ["learner_id", "num__impute__selected__cpo"]
        ]

        # ... the remaining must always have None set then
        for attr in attrs:
            assert attr is None

        if learner == "aknn":
            assert 18 <= self.aknn__M <= 50
            assert self.aknn__distance in ["l2", "cosine", "ip"]
            assert 7 <= self.aknn__ef <= 403
            assert 7 <= self.aknn__ef_construction <= 1097
            assert 1 <= self.aknn__k <= 50

        elif learner == "glmnet":
            assert 0.0 <= self.glmnet__alpha <= 1.0
            assert 0.0009118819655545162 <= self.glmnet__s <= 1096.6331584284585

        elif learner == "rpart":
            assert 0.0009118819655545162 <= self.rpart__cp <= 1.0
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

        elif learner == "svm":
            assert 4.5399929762484854e-05 <= self.svm__cost <= 22026.465794806718
            assert 4.5399929762484854e-05 <= self.svm__gamma <= 22026.465794806718
            assert self.svm__kernel in ["linear", "polynomial", "radial"]
            assert 4.5399929762484854e-05 <= self.svm__tolerance <= 2.0

            if self.svm__degree is not None:
                assert 2 <= self.svm__degree <= 5
                assert self.svm__kernel == "polynomial"

            if self.svm__gamma is not None:
                assert 4.5399929762484854e-05 <= self.svm__gamma <= 22026.465794806718
                assert self.svm__kernel == "radial"

        elif learner == "xgboost":
            assert self.xgboost__booster in ["gblinear", "gbtree", "dart"]
            assert 0.0009118819655545162 <= self.xgboost__alpha <= 1096.6331584284585
            assert 0.0009118819655545162 <= self.xgboost__lambda <= 1096.6331584284585
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
                assert 0.0009118819655545162 <= self.xgboost__eta <= 1.0

            if self.xgboost__gamma is not None:
                assert self.xgboost__booster in ["dart", "gbtree"]
                assert 4.5399929762484854e-05 <= self.xgboost__gamma <= 7.38905609893065

            if self.xgboost__max_depth is not None:
                assert self.xgboost__booster in ["dart", "gbtree"]
                assert 1 <= self.xgboost__max_depth <= 15

            if self.xgboost__min_child_weight is not None:
                assert self.xgboost__booster in ["dart", "gbtree"]
                assert (
                    2.718281828459045
                    <= self.xgboost__min_child_weight
                    <= 148.4131591025766
                )

            if self.xgboost__rate_drop is not None:
                assert self.xgboost__booster in ["dart"]
                assert 0.0 <= self.xgboost__rate_drop <= 1.0

            if self.xgboost__skip_drop is not None:
                assert self.xgboost__booster in ["dart"]
                assert 0.0 <= self.xgboost__skip_drop <= 1.0

        else:
            raise NotImplementedError()


@dataclass(frozen=True)
class RBV2SuperResult(RBV2Result):
    config: RBV2SuperConfig


class RBV2SuperBenchmark(RBV2Benchmark):
    name = "rbv2_super"

    Result = RBV2SuperResult
    Config = RBV2SuperConfig

    has_conditionals = True

    instances = [
        "41138",
        "40981",
        "4134",
        "1220",
        "4154",
        "41163",
        "4538",
        "40978",
        "375",
        "1111",
        "40496",
        "40966",
        "4534",
        "40900",
        "40536",
        "41156",
        "1590",
        "1457",
        "458",
        "469",
        "41157",
        "11",
        "1461",
        "1462",
        "1464",
        "15",
        "40975",
        "41142",
        "40701",
        "40994",
        "23",
        "1468",
        "40668",
        "29",
        "31",
        "6332",
        "37",
        "40670",
        "23381",
        "151",
        "188",
        "41164",
        "1475",
        "1476",
        "1478",
        "41169",
        "1479",
        "41212",
        "1480",
        "300",
        "41143",
        "1053",
        "41027",
        "1067",
        "1063",
        "41162",
        "3",
        "6",
        "1485",
        "1056",
        "12",
        "14",
        "16",
        "18",
        "40979",
        "22",
        "1515",
        "334",
        "24",
        "1486",
        "1493",
        "28",
        "1487",
        "1068",
        "1050",
        "1049",
        "32",
        "1489",
        "470",
        "1494",
        "182",
        "312",
        "40984",
        "1501",
        "40685",
        "38",
        "42",
        "44",
        "46",
        "40982",
        "1040",
        "41146",
        "377",
        "40499",
        "50",
        "54",
        "307",
        "1497",
        "60",
        "1510",
        "40983",
        "40498",
        "181",
    ]
