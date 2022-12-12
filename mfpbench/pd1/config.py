from __future__ import annotations

from dataclasses import dataclass

from mfpbench.config import Config


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class PD1Config(Config):
    """The config for PD1."""

    lr_decay_factor: float
    lr_initial: float
    lr_power: float
    opt_momentum: float
