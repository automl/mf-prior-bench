from dataclasses import dataclass

from mfpbench.config import Config


@dataclass(frozen=True, eq=False)  # type: ignore[misc]
class YAHPOConfig(Config):
    ...
