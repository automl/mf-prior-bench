from __future__ import annotations

from dataclasses import dataclass

from mfpbench.jahs.config import JAHSConfig
from mfpbench.result import Result


@dataclass
class JAHSResult(Result[JAHSConfig, int]):
    epoch: int

    # Info
    size: float  # MB
    flops: float
    latency: float  # unit?
    runtime: float  # unit?

    # Scores
    valid_acc: float
    test_acc: float
    train_acc: float

    @classmethod
    def from_dict(
        cls,
        config: JAHSConfig,
        result: dict,
        fidelity: int,
    ) -> JAHSResult:
        """

        Parameters
        ----------
        config: JAHSConfig
            The config used to generate these results

        result : dict
            The results to pull from

        fidelity : int
            The fidelity at which this config was evaluated, epochs

        Returns
        -------
        JAHSResult
        """
        return JAHSResult(
            config=config,
            epoch=fidelity,
            size=result["size_MB"],
            flops=result["FLOPS"],
            latency=result["latency"],
            runtime=result["runtime"],
            valid_acc=result["valid-acc"],
            test_acc=result["test-acc"],
            train_acc=result["train-acc"],
        )

    @property
    def score(self) -> float:
        """The score of interest"""
        return self.test_acc

    @property
    def fidelity(self) -> int:
        """The fidelity used"""
        return self.epoch
