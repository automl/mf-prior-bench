from __future__ import annotations
from typing import TypeVar

from dataclasses import dataclass

from mfpbench.result import Result
from mfpbench.yahpo.config import YAHPOConfig

# A YahpoResult is parametrized by a YAHPOConfig and fidelity type
C = TypeVar("C", bound=YAHPOConfig)
F = TypeVar("F", int, float)


@dataclass  # type: ignore[misc]
class YAHPOResult(Result[C, F]):
    ...
