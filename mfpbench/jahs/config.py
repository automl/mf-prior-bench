from __future__ import annotations

from dataclasses import dataclass

from mfpbench.config import Config


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class JAHSConfig(Config):
    """The config for JAHSBench, useful to have regardless of the configspace used.

    https://github.com/automl/jahs_bench_201/blob/main/jahs_bench/lib/core/configspace.py
    """

    # Not fidelities for our use case
    N: int
    W: int

    # Categoricals
    Op1: int
    Op2: int
    Op3: int
    Op4: int
    Op5: int
    Op6: int
    TrivialAugment: bool
    Activation: str
    Optimizer: str

    # Continuous Numericals
    Resolution: float
    LearningRate: float
    WeightDecay: float

    def validate(self) -> None:
        """Validate this config incase required."""
        # Just being explicit to catch bugs easily, we can remove later
        assert self.N in [1, 3, 5]
        assert self.W in [4, 8, 16]
        assert self.Op1 in [0, 1, 2, 3, 4, 5]
        assert self.Op2 in [0, 1, 2, 3, 4, 5]
        assert self.Op3 in [0, 1, 2, 3, 4, 5]
        assert self.Op4 in [0, 1, 2, 3, 4, 5]
        assert self.Op5 in [0, 1, 2, 3, 4, 5]
        assert self.Op6 in [0, 1, 2, 3, 4, 5]
        assert self.Resolution in [0.25, 0.5, 1.0]
        assert isinstance(self.TrivialAugment, bool)
        assert self.Activation in ["ReLU", "Hardswish", "Mish"]
        assert self.Optimizer in ["SGD"]
        assert 1e-3 <= self.LearningRate <= 1e0
        assert 1e-5 <= self.WeightDecay <= 1e-2
