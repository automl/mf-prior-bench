from typing import Callable

from mfpbench.download import Source, sources
from pytest import fixture


@fixture
def get_source() -> Callable[[str], Source]:
    def _get(name: str) -> Source:
        if name not in sources:
            raise ValueError(f"Not a known source, must be in {sources.keys()}")

        source = sources[name]

        if not source.exists():
            raise RuntimeError(
                f"{source} has not been installed yet, please run:\n"
                "python -m mfbench.download"
            )

        return source

    return _get
