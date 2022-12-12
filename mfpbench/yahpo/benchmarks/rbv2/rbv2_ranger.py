from __future__ import annotations

from dataclasses import dataclass
from typing import no_type_check

from typing_extensions import Literal

from mfpbench.yahpo.benchmarks.rbv2.rbv2 import RBV2Benchmark, RBV2Config, RBV2Result


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class RBV2rangerConfig(RBV2Config):

    num__impute__selected__cpo: Literal["impute.mean", "impute.median", "impute.hist"]

    min__node__size: int  # (1, 100)
    mtry__power: int  # (0, 1)
    num__trees: int  # (1, 2000)
    respect__unordered__factors: Literal["ignore", "order", "partition"]
    sample__fraction: float  # (0.1, 1.0)
    splitrule: Literal["gini", "extratrees"]

    num__random__splits: int | None = None  # (1, 100)

    @no_type_check
    def validate(self) -> None:
        """Validate this config."""
        assert self.num__impute__selected__cpo in [
            "impute.mean",
            "impute.median",
            "impute.hist",
        ]
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
class RBV2rangerResult(RBV2Result):
    config: RBV2rangerConfig


class RBV2rangerBenchmark(RBV2Benchmark):
    name = "rbv2_ranger"
    Result = RBV2rangerResult
    Config = RBV2rangerConfig
    has_conditionals = True

    instances = [
        "4135",
        "40981",
        "4134",
        "1220",
        "4154",
        "4538",
        "40978",
        "375",
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
        "1479",
        "41212",
        "1480",
        "41143",
        "1053",
        "41027",
        "1067",
        "1063",
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
        "41278",
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
        "41216",
        "307",
        "1497",
        "60",
        "1510",
        "40983",
        "40498",
        "181",
        "41138",
        "41163",
        "1111",
        "41159",
        "300",
        "41162",
        "23517",
        "41165",
        "4541",
        "41161",
        "41166",
        "40927",
        "41150",
        "23512",
        "41168",
        "1493",
        "40996",
        "554",
        "40923",
        "41169",
    ]
