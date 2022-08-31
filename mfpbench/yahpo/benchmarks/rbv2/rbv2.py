from __future__ import annotations

from typing import Any, Mapping, TypeVar

from dataclasses import dataclass

from mfpbench.yahpo.benchmark import YAHPOBenchmark
from mfpbench.yahpo.config import YAHPOConfig
from mfpbench.yahpo.result import YAHPOResult

C = TypeVar("C", bound="RBV2Config")
R = TypeVar("R", bound="RBV2Result")


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class RBV2Config(YAHPOConfig):
    ...


@dataclass(frozen=True)  # type: ignore[misc]
class RBV2Result(YAHPOResult):
    # Fidelity
    trainsize: float

    acc: float
    bac: float
    auc: float
    brier: float
    f1: float
    logloss: float

    timetrain: float
    timepredict: float

    memory: float

    @classmethod
    def from_dict(
        cls: type[RBV2Result],
        config: C,
        result: Mapping[str, Any],
        fidelity: float,
    ) -> RBV2Result:
        """

        Parameters
        ----------
        config: RBV2Config
            The config used to generate these results

        result : dict
            The results to pull from

        fidelity : float
            The fidelity at which this config was evaluated, epochs

        Returns
        -------
        RBV2Result
        """
        return RBV2Result(trainsize=fidelity, config=config, **result)

    @property
    def score(self) -> float:
        """The score of interest"""
        return self.bac

    @property
    def error(self) -> float:
        """The error of interest"""
        return 1 - self.bac

    @property
    def test_score(self) -> float:
        """The score on the test set"""
        return self.score

    @property
    def test_error(self) -> float:
        """The error on the test set"""
        return self.error

    @property
    def val_score(self) -> float:
        """The score on the validation set"""
        return self.score

    @property
    def val_error(self) -> float:
        """The error on the validation set"""
        return self.error

    @property
    def fidelity(self) -> float:
        """The fidelity used"""
        return self.trainsize

    @property
    def cost(self) -> float:
        """The time taken in seconds to train the config"""
        return self.train_time


class RBV2Benchmark(YAHPOBenchmark):
    # RVB2 class of benchmarks share train size as fidelity
    fidelity_range = (0.03, 1.0, 0.05)
    fidelity_name = "trainsize"
    _task_id_name = "task_id"

    # We have to specify a repl number, not sure what it is but YAHPO gym fix it to 10
    _forced_hps = {"repl": 10}
