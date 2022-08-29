from __future__ import annotations

from typing import Any, Mapping, TypeVar

from dataclasses import asdict, dataclass

from mfpbench.config import Config

Self = TypeVar("Self", bound="YAHPOConfig")


@dataclass(frozen=True, eq=False)  # type: ignore[misc]
class YAHPOConfig(Config):
    @classmethod
    def from_dict(cls: type[Self], d: Mapping[str, Any]) -> Self:
        """Create from a dict or mapping object"""
        # We may have keys that are conditional and hence we need to flatten them
        config = {k.replace(".", "__"): v for k, v in d.items()}
        return cls(**config)

    def dict(self) -> dict[str, Any]:
        """Converts the config to a raw dictionary"""
        d = asdict(self)
        return {k.replace("__", "."): v for k, v in d.items()}
