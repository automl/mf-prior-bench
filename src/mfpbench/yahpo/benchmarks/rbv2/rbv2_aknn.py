from __future__ import annotations

from dataclasses import dataclass
from typing import no_type_check
from typing_extensions import Literal

from mfpbench.yahpo.benchmarks.rbv2.rbv2 import RBV2Benchmark, RBV2Config, RBV2Result


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class RBV2aknnConfig(RBV2Config):
    num__impute__selected__cpo: Literal["impute.mean", "impute.median", "impute.hist"]
    M: int  # (18, 50)
    distance: Literal["l2", "cosine", "ip"]
    ef: int  # (7, 403),  log
    ef_construction: int  # (7, 1097),  log
    k: int  # (1, 50)

    @no_type_check
    def validate(self) -> None:
        """Validate this config."""
        assert self.num__impute__selected__cpo in [
            "impute.mean",
            "impute.median",
            "impute.hist",
        ]
        assert 18 <= self.M <= 50
        assert self.distance in ["l2", "cosine", "ip"]
        assert 7 <= self.ef <= 403
        assert 7 <= self.ef_construction <= 1097
        assert 1 <= self.k <= 50


@dataclass(frozen=True)
class RBV2aknnResult(RBV2Result):
    config: RBV2aknnConfig


class RBV2aknnBenchmark(RBV2Benchmark):
    Result = RBV2aknnResult
    Config = RBV2aknnConfig
    has_conditionals = False

    yahpo_base_benchmark_name = "rbv2_aknn"
    yahpo_instances = (
        "41138",
        "40981",
        "4134",
        "40927",
        "1220",
        "4154",
        "41163",
        "40996",
        "4538",
        "40978",
        "375",
        "1111",
        "40496",
        "40966",
        "41150",
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
        "4541",
        "40670",
        "23381",
        "151",
        "188",
        "41164",
        "1475",
        "1476",
        "41159",
        "1478",
        "41169",
        "23512",
        "1479",
        "41212",
        "1480",
        "300",
        "41168",
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
        "554",
        "334",
        "24",
        "1486",
        "23517",
        "41278",
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
        "41161",
        "41165",
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
        "41166",
        "307",
        "1497",
        "60",
        "1510",
        "40983",
        "40498",
        "181",
        "40923",
    )