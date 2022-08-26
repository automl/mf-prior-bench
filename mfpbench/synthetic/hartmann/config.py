from dataclasses import dataclass

from mfpbench.config import Config


@dataclass(frozen=True, eq=False)  # type: ignore[misc]
class MFHartmannConfig(Config):
    ...


@dataclass(frozen=True, eq=False)  # type: ignore[misc]
class MFHartmann3Config(MFHartmannConfig):
    X_0: float
    X_1: float
    X_2: float

    def validate(self) -> None:
        """Validate this config"""
        assert isinstance(self.X_0, float)
        assert isinstance(self.X_1, float)
        assert isinstance(self.X_2, float)


@dataclass(frozen=True, eq=False)  # type: ignore[misc]
class MFHartmann6Config(MFHartmannConfig):
    X_0: float
    X_1: float
    X_2: float
    X_3: float
    X_4: float
    X_5: float

    def validate(self) -> None:
        """Validate this config"""
        assert isinstance(self.X_0, float)
        assert isinstance(self.X_1, float)
        assert isinstance(self.X_2, float)
        assert isinstance(self.X_3, float)
        assert isinstance(self.X_4, float)
        assert isinstance(self.X_5, float)
