from __future__ import annotations

from typing import Any, Mapping, TypeVar

from dataclasses import asdict, dataclass

from mfpbench.config import Config

Self = TypeVar("Self", bound="YAHPOConfig")


@dataclass(frozen=True, eq=False)  # type: ignore[misc]
class YAHPOConfig(Config):
    ...
