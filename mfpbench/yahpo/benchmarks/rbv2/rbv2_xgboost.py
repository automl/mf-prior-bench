from __future__ import annotations

from dataclasses import dataclass
from typing import no_type_check

from typing_extensions import Literal

from mfpbench.yahpo.benchmarks.rbv2.rbv2 import RBV2Benchmark, RBV2Config, RBV2Result


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class RBV2xgboostConfig(RBV2Config):

    num__impute__selected__cpo: Literal["impute.mean", "impute.median", "impute.hist"]
    nrounds: int  # (7, 2981), log
    subsample: float  # (0.1, 1.0)
    alpha: float  # (0.0009118819655545162, 1096.6331584284585), log
    _lambda: float  # (0.0009118819655545162, 1096.6331584284585), log
    booster: Literal["gblinear", "gbtree", "dart"]

    colsample_bylevel: float | None = None  # (0.01, 1.0)
    colsample_bytree: float | None = None  # (0.01, 1.0)
    eta: float | None = None  # (0.0009118819655545162, 1.0)  log
    # (4.5399929762484854e-05, 7.38905609893065), log
    gamma: float | None = None
    max_depth: int | None = None  # (1, 15)
    # (2.718281828459045, 148.4131591025766),  log
    min_child_weight: float | None = None
    rate_drop: float | None = None  # (0.0, 1.0)
    skip_drop: float | None = None  # (0.0, 1.0)

    @no_type_check
    def validate(self) -> None:
        """Validate this config."""
        assert self.booster in ["gblinear", "gbtree", "dart"]
        assert 0.0009118819655545162 <= self.alpha <= 1096.6331584284585
        assert 0.0009118819655545162 <= self._lambda <= 1096.6331584284585
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
            assert 0.0009118819655545162 <= self.eta <= 1.0

        if self.gamma is not None:
            assert self.booster in ["dart", "gbtree"]
            assert 4.5399929762484854e-05 <= self.gamma <= 7.38905609893065

        if self.max_depth is not None:
            assert self.booster in ["dart", "gbtree"]
            assert 1 <= self.max_depth <= 15

        if self.min_child_weight is not None:
            assert self.booster in ["dart", "gbtree"]
            assert 2.718281828459045 <= self.min_child_weight <= 148.4131591025766

        if self.rate_drop is not None:
            assert self.booster in ["dart"]
            assert 0.0 <= self.rate_drop <= 1.0

        if self.skip_drop is not None:
            assert self.booster in ["dart"]
            assert 0.0 <= self.skip_drop <= 1.0

        assert self.num__impute__selected__cpo in [
            "impute.mean",
            "impute.median",
            "impute.hist",
        ]


@dataclass(frozen=True)
class RBV2xgboostResult(RBV2Result):
    config: RBV2xgboostConfig


class RBV2xgboostBenchmark(RBV2Benchmark):
    name = "rbv2_xgboost"
    Result = RBV2xgboostResult
    Config = RBV2xgboostConfig
    has_conditionals = True

    _replacements_hps = [("_lambda", "lambda")]

    instances = [
        "16",
        "40923",
        "41143",
        "470",
        "1487",
        "40499",
        "40966",
        "41164",
        "1497",
        "40975",
        "1461",
        "41278",
        "11",
        "54",
        "300",
        "40984",
        "31",
        "1067",
        "1590",
        "40983",
        "41163",
        "41165",
        "182",
        "1220",
        "41159",
        "41169",
        "42",
        "188",
        "1457",
        "1480",
        "6332",
        "181",
        "1479",
        "40670",
        "40536",
        "41138",
        "41166",
        "6",
        "14",
        "29",
        "458",
        "1056",
        "1462",
        "1494",
        "40701",
        "12",
        "1493",
        "44",
        "307",
        "334",
        "40982",
        "41142",
        "38",
        "1050",
        "469",
        "23381",
        "41157",
        "15",
        "4541",
        "23",
        "4134",
        "40927",
        "40981",
        "41156",
        "3",
        "1049",
        "40900",
        "1063",
        "23512",
        "40979",
        "1040",
        "1068",
        "41161",
        "22",
        "1489",
        "41027",
        "24",
        "4135",
        "23517",
        "1053",
        "1468",
        "312",
        "377",
        "1515",
        "18",
        "1476",
        "1510",
        "41162",
        "28",
        "375",
        "1464",
        "40685",
        "40996",
        "41146",
        "41216",
        "40668",
        "41212",
        "32",
        "60",
        "4538",
        "40496",
        "41150",
        "37",
        "46",
        "554",
        "1475",
        "1485",
        "1501",
        "1111",
        "4534",
        "41168",
        "151",
        "4154",
        "40978",
        "40994",
        "50",
        "1478",
        "1486",
        "40498",
    ]
