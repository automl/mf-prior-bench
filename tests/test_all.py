from __future__ import annotations

from typing import Any

from pathlib import Path

import numpy as np
import pytest
from pytest_cases import fixture, parametrize

import mfpbench
from mfpbench import Benchmark, MFHartmannBenchmark, YAHPOBenchmark

SEED = 1
CONDITONALS = False  # We currently can't do these

# Edit this if you have data elsewhere
HERE = Path(__file__).parent.resolve()
DATADIR: Path | None = None

# We can get all the benchmarks here
available_benchmarks = [
    (name, params) for name, _, params in mfpbench.available(conditionals=CONDITONALS)
]


# We expect the default download location for each
@fixture(scope="module")
@parametrize(item=available_benchmarks)
def benchmark(item: tuple[str, dict[str, Any] | None]) -> Benchmark:
    """The JAHSBench series of benchmarks"""
    name, params = item
    if params is None:
        params = {}

    if DATADIR is None:
        benchmark = mfpbench.get(name=name, seed=SEED, **params)
    else:
        benchmark = mfpbench.get(name=name, seed=SEED, datadir=DATADIR, **params)

    # We force benchmarks to load if they must
    benchmark.load()

    return benchmark


@parametrize("n_samples", [1, 10, 100])
def test_benchmark_sampling(benchmark: Benchmark, n_samples: int) -> None:
    """
    Expects
    -------
    * Can sample 1 or many configs
    * They are all of type benchmark.Config
    * All sampled configs are valid
    """
    config = benchmark.sample()
    assert isinstance(config, benchmark.Config)
    config.validate()

    configs = benchmark.sample(n_samples)
    assert len(configs) == n_samples
    assert all(isinstance(config, benchmark.Config) for config in configs)

    for config in configs:
        config.validate()


def test_query_api_validity(benchmark: Benchmark) -> None:
    """
    Expects
    -------
    * Can query a sampled config
    * Can query from a Configuration from config space
    * Can query with the dict version of either of the above
    """
    sample = benchmark.sample()
    result = benchmark.query(sample)
    assert result.config == sample

    sample_dict = sample.dict()
    result = benchmark.query(sample_dict)
    assert result.config == sample_dict

    configspace_sample = benchmark.space.sample_configuration()
    result = benchmark.query(configspace_sample)
    assert result.config == configspace_sample

    configspace_sample_dict = {**configspace_sample}
    result = benchmark.query(configspace_sample_dict)
    assert result.config == configspace_sample_dict


def test_result_api_validity(benchmark: Benchmark) -> None:
    """
    Expects
    -------
    * Can get all relevant metrics
    * Can query from a Configuration from config space
    * Can query with the dict version of either of the above
    """
    sample = benchmark.sample()
    result = benchmark.query(sample)

    assert result.score is not None
    assert result.error is not None
    assert result.fidelity is not None

    if isinstance(benchmark, MFHartmannBenchmark):
        # These don't make sense for a synthetic benchmark which doesn't
        # train anything
        assert result.val_score is not None
        assert result.test_score is not None
        assert result.train_time is not None


def test_query_through_entire_fidelity_range(benchmark: Benchmark) -> None:
    """
    Expects
    -------
    * Should be able to query the benchmark over the entire fidelity range
    * All results should have their fidelity between the start and step
    """
    config = benchmark.sample()

    results = [benchmark.query(config, at=x) for x in benchmark.iter_fidelities()]
    start, stop, _ = benchmark.fidelity_range
    for r in results:
        assert start <= r.fidelity <= stop
        assert r.config == config


def test_repeated_query(benchmark: Benchmark) -> None:
    """
    Expects
    -------
    * Repeating the same query multiple times will give the same results
    """
    configs = benchmark.sample(10)
    for f in benchmark.iter_fidelities():

        results1 = [benchmark.query(config, at=f) for config in configs]
        results2 = [benchmark.query(config, at=f) for config in configs]

        for r1, r2 in zip(results1, results2):
            assert r1 == r2, f"{r1}\n{r2}"


def test_repeated_trajectory(benchmark: Benchmark) -> None:
    """
    Expects
    -------
    * Repeating the same trajectory multiple times will give the same results
    """
    configs = benchmark.sample(10)

    for config in configs:

        results = np.asarray(
            [[benchmark.trajectory(config)] for _ in range(3)],
            dtype=object,
        )

        for row in results.T:
            first = row[0]
            for other in row[1:]:
                assert first == other


def test_query_default_is_max_fidelity(benchmark: Benchmark) -> None:
    """
    Expects
    -------
    * A query with no args is the same as using the max fidelity directly
    """
    config = benchmark.sample()
    r1 = benchmark.query(config, at=benchmark.end)
    r2 = benchmark.query(config)

    assert r1 == r2


def test_query_same_as_trajectory(benchmark: Benchmark) -> None:
    """
    Expects
    -------
    * Querying every point individually should be the same as using trajectory
    """
    config = benchmark.sample()
    if isinstance(benchmark, YAHPOBenchmark):
        pytest.skip(
            "YAHPOBench gives slight numerical instability when querying in bulk vs"
            " each config individually."
        )

    query_results = [benchmark.query(config, at=f) for f in benchmark.iter_fidelities()]
    trajectory_results = benchmark.trajectory(config)

    for qr, tr in zip(query_results, trajectory_results):
        assert qr == tr, f"{qr}\n{tr}"


def test_trajectory_is_over_full_range_by_default(benchmark: Benchmark) -> None:
    """
    Expects
    -------
    * The trajectory should by default go from the start fidelity to the max
    """
    config = benchmark.sample()
    results = benchmark.trajectory(config)

    for r, fidelity in zip(results, benchmark.iter_fidelities()):
        assert r.fidelity == fidelity
