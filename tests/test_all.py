from __future__ import annotations

from pytest_cases import fixture, parametrize

import mfpbench
from mfpbench import Benchmark, JAHSBenchmark

SEED = 1

# We can get all the benchmarks her
available_benchmarks = [(name, task_id) for name, _, task_id in mfpbench.available()]


# We expect the default download location for each
@fixture(scope="module")
@parametrize(item=available_benchmarks)
def benchmark(item: tuple[str, str | None]) -> Benchmark:
    """The JAHSBench series of benchmarks"""
    name, task_id = item
    benchmark = mfpbench.get(name=name, task_id=task_id, seed=SEED)

    # We force JAHSBenchmark ones to load itself before the test
    if isinstance(benchmark, JAHSBenchmark):
        print(f"Initilizing {name}")
        benchmark.bench

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
