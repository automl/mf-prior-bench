from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mfpbench
import numpy as np
import pytest
from mfpbench import (
    Benchmark,
    IAMLglmnetBenchmark,
    JAHSCifar10,
    LCBenchBenchmark,
    MFHartmann3BenchmarkBad,
    MFHartmann3BenchmarkGood,
    MFHartmann3BenchmarkTerrible,
    MFHartmann6BenchmarkBad,
    MFHartmann6BenchmarkGood,
    MFHartmann6BenchmarkTerrible,
    PD1lm1b_transformer_2048,
    PD1translatewmt_xformer_64,
    PD1uniref50_transformer_128,
    RBV2aknnBenchmark,
    YAHPOBenchmark,
)
from pytest_cases import fixture, parametrize

SEED = 1
CONDITONALS = False  # We currently can't do these
AVAILABLE_PRIORS = ["good", "medium", "bad"]

# Edit this if you have data elsewhere
HERE = Path(__file__).parent.resolve()
DATADIR: Path | None = None


@dataclass
class BenchmarkTest:
    name: str
    cls: type[Benchmark]
    prior: str | None = None
    kwargs: dict[str, Any] | None = None

    def unpack(self) -> dict[str, Any]:
        p: dict[str, Any] = {
            "name": self.name,
            "prior": self.prior,
            "seed": SEED,
        }
        if self.kwargs:
            p.update(self.kwargs)
        if DATADIR is not None:
            p["datadir"] = DATADIR
        return p


# List of benchmarks we care to test
benchmarks = [
    BenchmarkTest("jahs_cifar10", JAHSCifar10),
    BenchmarkTest("jahs_cifar10", JAHSCifar10, prior="good"),
    #
    BenchmarkTest("mfh3_good", MFHartmann3BenchmarkGood, prior="perfect"),
    BenchmarkTest("mfh3_terrible", MFHartmann3BenchmarkTerrible, prior="perfect"),
    BenchmarkTest("mfh3_bad", MFHartmann3BenchmarkBad),
    #
    BenchmarkTest("mfh6_good", MFHartmann6BenchmarkGood, prior="perfect"),
    BenchmarkTest("mfh6_terrible", MFHartmann6BenchmarkTerrible, prior="perfect"),
    BenchmarkTest("mfh6_bad", MFHartmann6BenchmarkBad),
    #
    BenchmarkTest(
        "lcbench",
        LCBenchBenchmark,
        kwargs={"task_id": LCBenchBenchmark.instances[0]},  # type: ignore
    ),
    BenchmarkTest(
        "rbv2_aknn",
        RBV2aknnBenchmark,
        kwargs={"task_id": RBV2aknnBenchmark.instances[0]},  # type: ignore
    ),
    BenchmarkTest(
        "iaml_glmnet",
        IAMLglmnetBenchmark,
        kwargs={"task_id": IAMLglmnetBenchmark.instances[0]},  # type: ignore
    ),
    #
    BenchmarkTest("lm1b_transformer_2048", PD1lm1b_transformer_2048),
    BenchmarkTest("translatewmt_xformer_64", PD1translatewmt_xformer_64),
    BenchmarkTest("uniref50_transformer_128", PD1uniref50_transformer_128),
]


# We expect the default download location for each
@fixture(scope="module")
@parametrize(item=benchmarks)
def benchmark(item: BenchmarkTest) -> Benchmark:
    """The JAHSBench series of benchmarks."""
    benchmark = mfpbench.get(**item.unpack())
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
    * All sampled configs are valid.
    """
    config = benchmark.sample()
    assert isinstance(config, benchmark.Config)
    config.validate()

    configs = benchmark.sample(n_samples)
    assert len(configs) == n_samples
    for config in configs:
        assert isinstance(config, benchmark.Config)

    for config in configs:
        config.validate()


def test_query_api_validity(benchmark: Benchmark) -> None:
    """
    Expects
    -------
    * Can query a sampled config
    * Can query from a Configuration from config space
    * Can query with the dict version of either of the above.
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
    * Can query with the dict version of either of the above.
    """
    sample = benchmark.sample()
    result = benchmark.query(sample)

    assert result.score is not None
    assert result.error is not None
    assert result.test_score is not None
    assert result.test_error is not None
    assert result.val_score is not None
    assert result.val_error is not None
    assert result.fidelity is not None
    assert result.cost is not None


def test_query_through_entire_fidelity_range(benchmark: Benchmark) -> None:
    """
    Expects
    -------
    * Should be able to query the benchmark over the entire fidelity range
    * All results should have their fidelity between the start and step.
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
    * Repeating the same query multiple times will give the same results.
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
    * Repeating the same trajectory multiple times will give the same results.
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
    * A query with no args is the same as using the max fidelity directly.
    """
    config = benchmark.sample()
    r1 = benchmark.query(config, at=benchmark.end)
    r2 = benchmark.query(config)

    assert r1 == r2


def test_query_same_as_trajectory(benchmark: Benchmark) -> None:
    """
    Expects
    -------
    * Querying every point individually should be the same as using trajectory.
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
    * The trajectory should by default go from the start fidelity to the max.
    """
    config = benchmark.sample()
    results = benchmark.trajectory(config)

    for r, fidelity in zip(results, benchmark.iter_fidelities()):
        assert r.fidelity == fidelity


def test_configs_hashable_and_unique(benchmark: Benchmark) -> None:
    configs = benchmark.sample(10)

    s = set(configs)
    assert len(s) == len(configs)


def test_results_hashable_and_unique(benchmark: Benchmark) -> None:
    configs = benchmark.sample(10)
    results = [benchmark.query(c) for c in configs]

    s = set(results)
    assert len(s) == len(results)


def test_config_with_same_content_hashes_correctly(benchmark: Benchmark) -> None:
    config = benchmark.sample()

    # Turn it into a dict and back again
    new_config = benchmark.Config.from_dict(config.dict())

    assert hash(config) == hash(new_config)


def test_result_with_same_content_hashes_correctly(benchmark: Benchmark) -> None:
    config = benchmark.sample()
    result = benchmark.query(config)

    # Turn it into a dict and back again
    new_result = benchmark.Result(
        config=config,
        fidelity=result.fidelity,
        **result.dict(),
    )

    assert hash(result) == hash(new_result)


def test_result_same_value_but_different_fidelity_has_different_hash(
    benchmark: Benchmark,
) -> None:
    config = benchmark.sample()
    result = benchmark.query(config)

    # Turn it into a dict and back again
    new_result = benchmark.Result(
        config=config,
        fidelity=result.fidelity - 1,
        **result.dict(),
    )

    assert hash(result) != hash(new_result)


@parametrize(item=benchmarks)
def test_prior_from_available_priors(item: BenchmarkTest) -> None:
    """
    Expects
    -------
    * Getting a benchmark with an available prior should have its configspace seeded
      with that prior.
    """
    params = item.unpack()

    # Test begins
    # We seed it with all the priors advertised
    for prior in AVAILABLE_PRIORS:

        params["prior"] = prior
        bench = mfpbench.get(**params)

        # The default configuration for the benchmark should be the same as the prior
        prior_config = bench.prior
        default = bench.space.get_default_configuration()
        assert default == prior_config, f"{prior}, {prior_config}, {default}"


@parametrize(item=benchmarks)
def test_prior_from_yaml_file(item: BenchmarkTest, tmp_path: Path) -> None:
    """
    Expects
    -------
    * Using a prior from a yaml file will have the configspace for the benchmark seeded
    with that config.
    """
    params = item.unpack()
    bench = mfpbench.get(**params)

    # Get a random config and save it temporarily
    random_config = bench.sample()

    path = tmp_path / "config.yaml"
    random_config.save(path, format="yaml")

    # Use the path of the saved config as the prior config
    params["prior"] = path
    bench = mfpbench.get(**params)

    # The default configuration for the benchmark should be the same as the prior
    default = bench.space.get_default_configuration()
    assert default == random_config, f"{random_config}, {default}"


@parametrize(item=benchmarks)
def test_prior_from_json_file(item: BenchmarkTest, tmp_path: Path) -> None:
    """
    Expects
    -------
    * Using a prior from a json file will have the configspace for the benchmark seeded
    with that config.
    """
    params = item.unpack()
    bench = mfpbench.get(**params)

    # Get a random config and save it temporarily
    random_config = bench.sample()

    path = tmp_path / "config.json"
    random_config.save(path, format="json")

    # Use the path of the saved config as the prior config
    params["prior"] = path
    bench = mfpbench.get(**params)

    # The default configuration for the benchmark should be the same as the prior
    default = bench.space.get_default_configuration()
    assert default == random_config, f"{random_config}, {default}"


@parametrize(item=benchmarks)
def test_prior_from_config(item: BenchmarkTest) -> None:
    """
    Expects
    -------
    * Using a prior from a config will have the configspace for the benchmark seeded
    with that config.
    """
    params = item.unpack()
    bench = mfpbench.get(**params)

    # Get a random config
    random_config = bench.sample()

    # Use the path of the saved config as the prior config
    params["prior"] = random_config
    bench = mfpbench.get(**params)

    # The default configuration for the benchmark should be the same as the prior
    default = bench.space.get_default_configuration()
    assert default == random_config, f"{random_config}, {default}"


@parametrize(item=benchmarks)
def test_prior_from_configuration(item: BenchmarkTest) -> None:
    """
    Expects
    -------
    * Using a prior from a config will have the configspace for the benchmark seeded
    with that config.
    """
    params = item.unpack()

    bench = mfpbench.get(**params)

    # Get a random config
    random_config = bench.space.sample_configuration()

    # Use the path of the saved config as the prior config
    params["prior"] = random_config
    bench = mfpbench.get(**params)

    # The default configuration for the benchmark should be the same as the prior
    default = bench.space.get_default_configuration()
    default = bench.Config.from_configuration(default)
    assert default == random_config, f"{random_config}, {default}"


@parametrize(item=benchmarks)
def test_prior_from_dict(item: BenchmarkTest) -> None:
    """
    Expects
    -------
    * Using a prior from a config will have the configspace for the benchmark seeded
    with that config.
    """
    params = item.unpack()
    bench = mfpbench.get(**params)

    # Get a random config
    random_config = bench.sample()

    # Use the path of the saved config as the prior config
    params["prior"] = random_config.dict()
    bench = mfpbench.get(**params)

    # The default configuration for the benchmark should be the same as the prior
    default = bench.space.get_default_configuration()
    assert default == random_config, f"{random_config}, {default}"


@parametrize(item=benchmarks)
def test_argmax_query(item: BenchmarkTest) -> None:
    """
    Expects
    -------
    * The argmax query return the best objective value found from the trajectory
      up to a given fidelity.
    """
    params = item.unpack()
    bench = mfpbench.get(**params)

    # Get a random configuration
    random_config = bench.sample()

    # Get the argmax
    argmax_config = bench.query(random_config, argmax=True)

    # Get the trajectory
    trajectory = bench.trajectory(random_config)
    best_in_trajectory = max(trajectory, key=lambda x: x.score)

    assert argmax_config == best_in_trajectory
