from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, TypeVar

from mfpbench.yahpo.benchmark import YAHPOBenchmark
from mfpbench.yahpo.config import YAHPOConfig
from mfpbench.yahpo.result import YAHPOResult

C = TypeVar("C", bound="RBV2Config")
R = TypeVar("R", bound="RBV2Result")


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class RBV2Config(YAHPOConfig):
    @classmethod
    def from_dict(cls: type[C], d: Mapping[str, Any]) -> C:
        """Create from a dict or mapping object."""
        # We may have keys that are conditional and hence we need to flatten them
        config = {k.replace(".", "__"): v for k, v in d.items()}
        return cls(**config)

    def dict(self) -> dict[str, Any]:
        """Converts the config to a raw dictionary."""
        d = asdict(self)
        return {k.replace("__", "."): v for k, v in d.items() if v is not None}


@dataclass(frozen=True)  # type: ignore[misc]
class RBV2Result(YAHPOResult[C, float]):
    # Fidelity
    fidelity: float

    acc: float
    bac: float
    auc: float
    brier: float
    f1: float
    logloss: float

    timetrain: float
    timepredict: float

    memory: float

    @property
    def score(self) -> float:
        """The score of interest."""
        return self.bac

    @property
    def error(self) -> float:
        """The error of interest."""
        return 1 - self.bac

    @property
    def test_score(self) -> float:
        """The score on the test set."""
        return self.score

    @property
    def test_error(self) -> float:
        """The error on the test set."""
        return self.error

    @property
    def val_score(self) -> float:
        """The score on the validation set."""
        return self.score

    @property
    def val_error(self) -> float:
        """The error on the validation set."""
        return self.error

    @property
    def cost(self) -> float:
        """The time taken in seconds to train the config."""
        return self.timetrain


class RBV2Benchmark(YAHPOBenchmark):
    # RVB2 class of benchmarks share train size as fidelity
    fidelity_range = (0.03, 1.0, 0.05)
    fidelity_name = "trainsize"
    _task_id_name = "task_id"

    # We have to specify a repl number, not sure what it is but YAHPO gym fix it to 10
    _forced_hps = {"repl": 10}
