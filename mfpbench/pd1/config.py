from __future__ import annotations

from dataclasses import dataclass

from mfpbench.config import Config


@dataclass(frozen=True, eq=False, unsafe_hash=True)
class PD1Config(Config):
    """The config for PD1"""

    hp1: int
    hp2: int

    def validate(self) -> None:
        """Validate this config incase required"""
        # Just being explicit to catch bugs easily, we can remove later
        # TODO: Doesn't need to be here but it helps to build up an idea of possible
        # configs and validate we don't break it
        assert self.hp1 == 1
        assert self.hp2 == 1
