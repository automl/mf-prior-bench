from __future__ import annotations

from dataclasses import dataclass

from mfpbench.result import Result


@dataclass(frozen=True)  # type: ignore[misc]
class YAHPOResult(Result):
    ...
