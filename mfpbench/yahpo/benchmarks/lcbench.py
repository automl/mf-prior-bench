from __future__ import annotations

import warnings
from dataclasses import dataclass

from mfpbench.yahpo.benchmark import YAHPOBenchmark
from mfpbench.yahpo.config import YAHPOConfig
from mfpbench.yahpo.result import YAHPOResult


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class LCBenchConfig(YAHPOConfig):
    """A LCBench Config.

    Note:
    ----
    For ``momentum``, the paper seems to suggest it's (0.1, 0.9) but the configspace
    says (0.1, 0.99), going with the code version.
    """

    batch_size: int  # [16, 512] int log
    learning_rate: float  # [1e-04, 0.1] float log
    momentum: float  # [0.1, 0.99] float, see note above
    weight_decay: float  # [1e-5, 0.1] float
    num_layers: int  # [1, 5] int
    max_units: int  # [64, 1024] int log
    max_dropout: float  # [0.0, 1.0] float

    def validate(self) -> None:
        """Validate this is a correct config."""
        assert 16 <= self.batch_size <= 512
        assert 1e-04 <= self.learning_rate <= 0.1
        assert 0.1 <= self.momentum <= 0.99
        assert 1e-05 <= self.weight_decay <= 0.1
        assert 1 <= self.num_layers <= 5
        assert 64 <= self.max_units <= 1024
        assert 0.0 <= self.max_dropout <= 1.0


@dataclass(frozen=True)  # type: ignore[misc]
class LCBenchResult(YAHPOResult[LCBenchConfig, int]):
    time: float  # unit?

    val_accuracy: float
    val_cross_entropy: float
    val_balanced_accuracy: float

    test_cross_entropy: float
    test_balanced_accuracy: float

    @property
    def score(self) -> float:
        """The score of interest."""
        return self.val_balanced_accuracy

    @property
    def error(self) -> float:
        """The error of interest."""
        return 1 - self.val_balanced_accuracy

    @property
    def test_score(self) -> float:
        """The score on the test set."""
        return self.test_balanced_accuracy

    @property
    def test_error(self) -> float:
        """The score on the test set."""
        return 1 - self.test_balanced_accuracy

    @property
    def val_score(self) -> float:
        """The score on the validation set."""
        return self.val_balanced_accuracy

    @property
    def val_error(self) -> float:
        """The score on the validation set."""
        return 1 - self.val_balanced_accuracy

    @property
    def cost(self) -> float:
        """Time taken in seconds to train the config."""
        warnings.warn(f"Unsure of unit for `cost` on {self.__class__}")
        return self.time


class LCBenchBenchmark(YAHPOBenchmark):
    name = "lcbench"
    fidelity_name = "epoch"
    fidelity_range = (1, 52, 1)
    Config = LCBenchConfig
    Result = LCBenchResult
    _task_id_name = "OpenML_task_id"

    instances = [
        "3945",
        "7593",
        "34539",
        "126025",
        "126026",
        "126029",
        "146212",
        "167104",
        "167149",
        "167152",
        "167161",
        "167168",
        "167181",
        "167184",
        "167185",
        "167190",
        "167200",
        "167201",
        "168329",
        "168330",
        "168331",
        "168335",
        "168868",
        "168908",
        "168910",
        "189354",
        "189862",
        "189865",
        "189866",
        "189873",
        "189905",
        "189906",
        "189908",
        "189909",
    ]
