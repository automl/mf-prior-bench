from pytest_cases import fixture

from mfpbench.resultframe import ResultFrame
from mfpbench.synthetic.hartmann import (
    MFHartmann3BenchmarkGood,
    MFHartmann3Config,
    MFHartmannResult,
)

SEED = 1


@fixture(scope="module")
def bench() -> MFHartmann3BenchmarkGood:
    return MFHartmann3BenchmarkGood(seed=SEED)


def test_add(bench: MFHartmann3BenchmarkGood) -> None:
    """
    Expects
    -------
    * Should be able to add results individuall
    """
    samples = bench.sample(3)
    results = [bench.query(s) for s in samples]

    frame = bench.frame()
    for r in results:
        frame.add(r)

    assert len(frame) == 3


def test_extend(bench: MFHartmann3BenchmarkGood) -> None:
    """
    Expects
    -------
    * Should be able to add results individuall
    """
    samples = bench.sample(3)
    results = [bench.query(s) for s in samples]

    frame = bench.frame()
    frame.extend(results)

    assert len(frame) == 3


def test_getitem(bench: MFHartmann3BenchmarkGood) -> None:
    """
    Expects
    -------
    * Should be able to index by fidelity and get all results at that fidelity
    * Should be able to index by config and get all results for that config
    """
    config1, config2 = bench.sample(2)

    result11 = bench.query(config1, at=1)
    result12 = bench.query(config1, at=2)

    result21 = bench.query(config2, at=1)
    result22 = bench.query(config2, at=2)

    frame = bench.frame()
    frame.extend([result11, result12, result21, result22])

    assert frame[2] == [result12, result22]
    assert frame[config1] == [result11, result12]


def test_delitem(bench: MFHartmann3BenchmarkGood) -> None:
    config1, config2 = bench.sample(2)

    result11 = bench.query(config1, at=1)
    result12 = bench.query(config1, at=2)

    result21 = bench.query(config2, at=1)
    result22 = bench.query(config2, at=2)

    rf = ResultFrame()
    rf.extend([result11, result12, result21, result22])

    assert rf[2] == [result12, result22]
    assert rf[config1] == [result11, result12]
