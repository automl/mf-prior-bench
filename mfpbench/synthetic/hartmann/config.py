from dataclasses import dataclass

from mfpbench.config import Config


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class MFHartmannConfig(Config):
    ...


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class MFHartmann3Config(MFHartmannConfig):
    X_0: float
    X_1: float
    X_2: float

    def validate(self) -> None:
        """Validate this config."""
        assert 0.0 <= self.X_0 <= 1.0
        assert 0.0 <= self.X_1 <= 1.0
        assert 0.0 <= self.X_2 <= 1.0


@dataclass(frozen=True, eq=False, unsafe_hash=True)  # type: ignore[misc]
class MFHartmann6Config(MFHartmannConfig):
    X_0: float
    X_1: float
    X_2: float
    X_3: float
    X_4: float
    X_5: float

    def validate(self) -> None:
        """Validate this config."""
        assert 0.0 <= self.X_0 <= 1.0
        assert 0.0 <= self.X_1 <= 1.0
        assert 0.0 <= self.X_2 <= 1.0
        assert 0.0 <= self.X_3 <= 1.0
        assert 0.0 <= self.X_4 <= 1.0
        assert 0.0 <= self.X_5 <= 1.0
