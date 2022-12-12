from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Mapping, TypeVar, no_type_check

from typing_extensions import Literal

from mfpbench.yahpo.benchmark import YAHPOBenchmark
from mfpbench.yahpo.config import YAHPOConfig
from mfpbench.yahpo.result import YAHPOResult

Self = TypeVar("Self", bound="NB301Config")

ChoicesT = Literal[
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]

Choices = [
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]

_hp_name_extension = "NetworkSelectorDatasetInfo_COLON_darts_COLON_"


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class NB301Config(YAHPOConfig):

    edge_normal_0: ChoicesT
    edge_normal_1: ChoicesT

    edge_reduce_0: ChoicesT
    edge_reduce_1: ChoicesT

    inputs_node_reduce_3: Literal["0_1", "0_2", "1_2"]
    inputs_node_reduce_4: Literal["0_1", "0_2", "0_3", "1_2", "1_3", "2_3"]
    inputs_node_reduce_5: Literal[
        "0_1", "0_2", "0_3", "0_4", "1_2", "1_3", "1_4", "2_3", "2_4", "3_4"
    ]

    inputs_node_normal_3: Literal["0_1", "0_2", "1_2"]
    inputs_node_normal_4: Literal["0_1", "0_2", "0_3", "1_2", "1_3", "2_3"]
    inputs_node_normal_5: Literal[
        "0_1", "0_2", "0_3", "0_4", "1_2", "1_3", "1_4", "2_3", "2_4", "3_4"
    ]

    edge_normal_2: ChoicesT | None = None
    edge_normal_3: ChoicesT | None = None
    edge_normal_4: ChoicesT | None = None
    edge_normal_5: ChoicesT | None = None
    edge_normal_6: ChoicesT | None = None
    edge_normal_7: ChoicesT | None = None
    edge_normal_8: ChoicesT | None = None
    edge_normal_9: ChoicesT | None = None
    edge_normal_10: ChoicesT | None = None
    edge_normal_11: ChoicesT | None = None
    edge_normal_12: ChoicesT | None = None
    edge_normal_13: ChoicesT | None = None

    edge_reduce_2: ChoicesT | None = None
    edge_reduce_3: ChoicesT | None = None
    edge_reduce_4: ChoicesT | None = None
    edge_reduce_5: ChoicesT | None = None
    edge_reduce_6: ChoicesT | None = None
    edge_reduce_7: ChoicesT | None = None
    edge_reduce_8: ChoicesT | None = None
    edge_reduce_9: ChoicesT | None = None
    edge_reduce_10: ChoicesT | None = None
    edge_reduce_11: ChoicesT | None = None
    edge_reduce_12: ChoicesT | None = None
    edge_reduce_13: ChoicesT | None = None

    @no_type_check
    def validate(self) -> None:
        """Validate this is a correct config.

        Note:
        ----
        We don't check conditionals validity
        """
        nodes = list(range(13 + 1))
        cells = ["normal", "reduce"]
        for i, cell in product(nodes, cells):
            attr_name = f"edge_{cell}_{i}"
            attr = getattr(self, attr_name)
            assert attr is None or attr in Choices, attr_name

        choices_3 = ["0_1", "0_2", "1_2"]
        choices_4 = ["0_1", "0_2", "0_3", "1_2", "1_3", "2_3"]
        choices_5 = [
            "0_1",
            "0_2",
            "0_3",
            "0_4",
            "1_2",
            "1_3",
            "1_4",
            "2_3",
            "2_4",
            "3_4",
        ]

        nodes = list(range(3, 5 + 1))
        for i, choices in [(3, choices_3), (4, choices_4), (5, choices_5)]:
            normal_node = f"inputs_node_normal_{i}"
            assert getattr(self, normal_node) in choices

            reduce_node = f"inputs_node_reduce_{i}"
            assert getattr(self, reduce_node) in choices

    @classmethod
    def from_dict(cls: type[Self], d: Mapping[str, Any]) -> Self:
        """Create from a dict or mapping object."""
        # We just flatten things because it's way too big of a name
        config = {k.replace(_hp_name_extension, ""): v for k, v in d.items()}
        return cls(**config)

    def dict(self) -> dict[str, Any]:
        """Converts the config to a raw dictionary."""
        return {
            _hp_name_extension + k: v for k, v in asdict(self).items() if v is not None
        }


@dataclass(frozen=True)  # type: ignore[misc]
class NB301Result(YAHPOResult[NB301Config, int]):
    runtime: float  # unit?
    val_accuracy: float

    @property
    def score(self) -> float:
        """The score of interest."""
        return self.val_accuracy

    @property
    def error(self) -> float:
        """The error of interest."""
        return 1 - self.val_accuracy

    @property
    def test_score(self) -> float:
        """The score on the test set."""
        return self.val_accuracy

    @property
    def test_error(self) -> float:
        """The score on the test set."""
        return 1 - self.val_accuracy

    @property
    def val_score(self) -> float:
        """The score on the validation set."""
        return self.val_accuracy

    @property
    def val_error(self) -> float:
        """The score on the validation set."""
        return 1 - self.val_accuracy

    @property
    def cost(self) -> float:
        """Time taken in seconds to train the config."""
        warnings.warn(f"Unsure of unit for `cost` on {self.__class__}")
        return self.runtime


class NB301Benchmark(YAHPOBenchmark):
    name = "nb301"
    fidelity_name = "epoch"
    fidelity_range = (1, 98, 1)
    Config = NB301Config
    Result = NB301Result
    _task_id_name = None
    has_conditionals = True

    instances = ["CIFAR10"]
