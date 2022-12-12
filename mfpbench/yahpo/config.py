from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from mfpbench.config import Config

Self = TypeVar("Self", bound="YAHPOConfig")


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class YAHPOConfig(Config):
    ...
