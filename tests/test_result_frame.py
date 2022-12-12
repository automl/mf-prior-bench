from mfpbench.synthetic.hartmann import MFHartmann3BenchmarkGood
from pytest_cases import fixture

SEED = 1


@fixture(scope="module")
def bench() -> MFHartmann3BenchmarkGood:
    return MFHartmann3BenchmarkGood(seed=SEED)


def test_add(bench: MFHartmann3BenchmarkGood) -> None:
    """
    Expects
    -------
    * Should be able to add results individuall.
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
    * Should be able to add results individuall.
    """
    samples = bench.sample(3)
    results = [bench.query(s) for s in samples]

    frame = bench.frame()
    for result in results:
        frame.add(result)

    assert len(frame) == 3


def test_getitem(bench: MFHartmann3BenchmarkGood) -> None:
    """
    Expects
    -------
    * Should be able to index by fidelity and get all results at that fidelity
    * Should be able to index by config and get all results for that config.
    """
    config1, config2 = bench.sample(2)

    result11 = bench.query(config1, at=bench.start)
    result12 = bench.query(config1, at=bench.start + 1)

    result21 = bench.query(config2, at=bench.start)
    result22 = bench.query(config2, at=bench.start + 1)

    frame = bench.frame()
    for result in [result11, result12, result21, result22]:
        frame.add(result)

    assert frame[bench.start + 1] == [result12, result22]
    assert frame[config1] == [result11, result12]


def test_delitem(bench: MFHartmann3BenchmarkGood) -> None:
    config1, config2 = bench.sample(2)

    result11 = bench.query(config1, at=bench.start)
    result12 = bench.query(config1, at=bench.start + 1)

    result21 = bench.query(config2, at=bench.start)
    result22 = bench.query(config2, at=bench.start + 1)

    rf = bench.frame()
    for result in [result11, result12, result21, result22]:
        rf.add(result)

    assert rf[bench.start + 1] == [result12, result22]
    assert rf[config1] == [result11, result12]
