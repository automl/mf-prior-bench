from __future__ import annotations

from dataclasses import dataclass

from mfpbench.yahpo.benchmark import YAHPOBenchmark
from mfpbench.yahpo.config import YAHPOConfig
from mfpbench.yahpo.result import YAHPOResult


class LCBenchConfig(YAHPOConfig):
    batch_size: int  # [16, 512] int log
    learning_rate: float  # [1e-04, 0.1] float log
    momentum: float  # [0.1, 0.9] float
    weight_decay: float  # [1e-5, 0.1] float
    num_layers: int  # [1, 5] int
    max_units: int  # [64, 1024] int log
    max_dropout: float  # [0.0, 1.0] float

    def validate(self) -> None:
        """Validate this is a correct config"""
        assert 16 <= self.batch_size <= 512
        assert 1e-04 <= self.learning_rate <= 0.1
        assert 0.1 <= self.momentum <= 0.9
        assert 1e-05 <= self.weight_decay <= 0.1
        assert 1 <= self.num_layers <= 5
        assert 64 <= self.max_units <= 1024
        assert 0.0 <= self.max_dropout <= 1.0


@dataclass
class LCBenchResult(YAHPOResult[LCBenchConfig, int]):
    epoch: int

    time: float

    val_accuracy: float
    val_cross_entropy: float
    val_balanced_accuracy: float

    test_cross_entropy: float
    test_balanced_accuracy: float

    @classmethod
    def from_dict(
        cls: type[LCBenchResult],
        config: LCBenchConfig,
        result: dict,
        fidelity: int,
    ) -> LCBenchResult:
        """

        Parameters
        ----------
        config: LCBenchConfig
            The config used to generate these results

        result : dict
            The results to pull from

        fidelity : int
            The fidelity at which this config was evaluated, epochs

        Returns
        -------
        LCBenchResult
        """
        return LCBenchResult(epoch=fidelity, config=config, **result)

    @property
    def score(self) -> float:
        """The score of interest"""
        return self.test_balanced_accuracy

    @property
    def fidelity(self) -> int:
        """The fidelity used"""
        return self.epoch


class LCBenchBenchmark(YAHPOBenchmark[LCBenchConfig, LCBenchResult, int]):
    name: str = "lcbench"
    instances: list[str] = [
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
