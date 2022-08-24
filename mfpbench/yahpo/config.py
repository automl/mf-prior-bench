from dataclasses import dataclass

from mfpbench.config import Config


@dataclass(frozen=True)  # type: ignore[misc]
class YAHPOConfig(Config):
    ...
