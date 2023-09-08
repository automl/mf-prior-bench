from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from mfpbench.result import Result
from mfpbench.yahpo.config import YAHPOConfig

C = TypeVar("C", bound=YAHPOConfig)
F = TypeVar("F", int, float)


@dataclass(frozen=True)  # type: ignore[misc]
class YAHPOResult(Result[C, F]):
    ...
