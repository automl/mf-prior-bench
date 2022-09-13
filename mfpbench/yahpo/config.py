from __future__ import annotations

from typing import TypeVar

from dataclasses import dataclass

from mfpbench.config import Config

Self = TypeVar("Self", bound="YAHPOConfig")


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class YAHPOConfig(Config):
    ...
