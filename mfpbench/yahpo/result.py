from __future__ import annotations

from typing import TypeVar

from dataclasses import dataclass

from mfpbench.result import Result
from mfpbench.yahpo.config import YAHPOConfig

C = TypeVar("C", bound=YAHPOConfig)
F = TypeVar("F", int, float)


@dataclass(frozen=True)  # type: ignore[misc]
class YAHPOResult(Result[C, F]):
    ...
