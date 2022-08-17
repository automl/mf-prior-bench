from __future__ import annotations

from typing import Callable

from pytest import fixture

from mfpbench.download import JAHSBenchSource


@fixture
def jahs_source(get_source: Callable[[str], JAHSBenchSource]) -> JAHSBenchSource:
    return get_source("jahs-bench-data")


def test_jahs_data_downloaded(jahs_source: JAHSBenchSource) -> None:
    """
    Expects
    -------
    * Should have all data downloaded
    """
    assert jahs_source.exists()
