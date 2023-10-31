from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pytest_cases import (
    case,
    filters as ft,
    fixture,
    parametrize,
    parametrize_with_cases,
)

import mfpbench
from mfpbench import (
    Benchmark,
    GenericTabularBenchmark,
    MFHartmannBenchmark,
    TabularBenchmark,
    YAHPOBenchmark,
)
from mfpbench.setup_benchmark import download_status

CONDITONALS = False  # We currently can't do these

# Edit this if you have data elsewhere
HERE = Path(__file__).parent.resolve()
DATADIR: Path | None = None
SEED = 1


@dataclass
class BenchmarkTest:
    name: str
    prior: str | None = None
    benchmark: Benchmark | None = None
    kwargs: dict[str, Any] | None = None

    def unpack(self) -> dict[str, Any]:
        """Unpacks the benchmark into a dictionary."""
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


@case
@pytest.mark.skipif(
    download_status("jahs") is False,
    reason="jahs is not downloaded",
)
def case_jahs() -> BenchmarkTest:
    return BenchmarkTest("jahs", kwargs={"task_id": "CIFAR10"})


@case
@pytest.mark.skipif(
    download_status("yahpo") is False,
    reason="yahpo is not downloaded",
)
def case_lcbench_yahpo() -> BenchmarkTest:
    return BenchmarkTest("lcbench", kwargs={"task_id": "126026"})


@case
@pytest.mark.skipif(
    download_status("pd1") is False,
    reason="pd1 is not downloaded",
)
def case_pd1() -> BenchmarkTest:
    return BenchmarkTest("lm1b_transformer_2048")


@pytest.mark.skipif(
    download_status("lcbench-tabular") is False,
    reason="lcbench-tabular is not downloaded",
)
@case
def case_lcbench_tabular() -> BenchmarkTest:
    return BenchmarkTest("lcbench_tabular", kwargs={"task_id": "adult"})


@case
def case_mfh() -> BenchmarkTest:
    return BenchmarkTest("mfh3_good", prior="good")


@case(tags="generic_tabular")
def case_generic_tabular() -> BenchmarkTest:
    ids = "abcdefghijklmnopqrstuvwxyz"
    colors = ["red", "green", "blue"]
    shapes = ["circle", "square", "triangle"]
    animals = ["cat", "dog", "bird"]
    numbers = [1, 2, 3]
    floats = [1.0, 2.0, 3.0]
    config_values = product(colors, shapes, animals, numbers, floats)
    values = [
        pd.DataFrame(
            [
                {
                    "config": k,
                    "color": c,
                    "shape": s,
                    "animal": a,
                    "number": n,
                    "float": f,
                    "balanced_accuracy": v,
                    "fidelity": fid,
                }
                for fid, v in zip([10, 20, 30], [0.5, 0.6, 0.7])
            ],
        )
        for k, (c, s, a, n, f) in zip(ids, config_values)
    ]
    df = pd.concat(values, ignore_index=True)
    benchmark = GenericTabularBenchmark(
        df,
        name="testdata",
        id_key="config",
        fidelity_key="fidelity",
        config_keys=["color", "shape"],
        result_keys=["balanced_accuracy"],
        result_mapping={
            "error": lambda df: 1 - df["balanced_accuracy"],
            "val_error": lambda df: 1 - df["balanced_accuracy"],
            "test_error": lambda df: 1 - df["balanced_accuracy"],
            "score": lambda df: df["balanced_accuracy"],
            "val_score": lambda df: df["balanced_accuracy"],
            "test_score": lambda df: df["balanced_accuracy"],
            "cost": lambda df: df["float"],
        },
        remove_constants=True,
    )
    return BenchmarkTest(benchmark.name, benchmark=benchmark)


# We expect the default download location for each
@fixture(scope="module")
@parametrize_with_cases("item", cases=".")
def benchmark(item: BenchmarkTest) -> Benchmark:
    if item.benchmark is not None:
        return item.benchmark

    benchmark = mfpbench.get(**item.unpack())
    # We force benchmarks to load if they must
    benchmark.load()

    return benchmark


@parametrize("n_samples", [1, 2, 3])
def test_benchmark_sampling(benchmark: Benchmark, n_samples: int) -> None:
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
    sample = benchmark.sample()
    result = benchmark.query(sample)

    assert result.config == sample

    sample_dict = sample.dict()
    result = benchmark.query(sample_dict)
    assert result.config == sample_dict


def test_result_api_validity(benchmark: Benchmark) -> None:
    sample = benchmark.sample()
    result = benchmark.query(sample)

    # MFHartmanns don't have scores
    if not isinstance(benchmark, MFHartmannBenchmark):
        assert result.score is not None
        assert result.test_score is not None
        assert result.val_score is not None

    assert result.error is not None
    assert result.test_error is not None
    assert result.val_error is not None
    assert result.fidelity is not None
    assert result.cost is not None


def test_query_through_entire_fidelity_range(benchmark: Benchmark) -> None:
    config = benchmark.sample()

    results = [benchmark.query(config, at=x) for x in benchmark.iter_fidelities()]
    start, stop, _ = benchmark.fidelity_range
    for r in results:
        assert start <= r.fidelity <= stop
        assert r.config == config


def test_repeated_query(benchmark: Benchmark) -> None:
    configs = benchmark.sample(10)
    for f in benchmark.iter_fidelities():
        results1 = [benchmark.query(config, at=f) for config in configs]
        results2 = [benchmark.query(config, at=f) for config in configs]

        for r1, r2 in zip(results1, results2):
            assert r1 == r2, f"{r1}\n{r2}"


def test_repeated_trajectory(benchmark: Benchmark) -> None:
    configs = benchmark.sample(10)

    for config in configs:
        traj1 = benchmark.trajectory(config)
        traj2 = benchmark.trajectory(config)
        for r1, r2 in zip(traj1, traj2):
            assert r1 == r2, f"{r1}\n{r2}"


def test_query_default_is_max_fidelity(benchmark: Benchmark) -> None:
    config = benchmark.sample()
    r1 = benchmark.query(config, at=benchmark.end)
    r2 = benchmark.query(config)

    assert r1 == r2


def test_query_same_as_trajectory(benchmark: Benchmark) -> None:
    config = benchmark.sample()
    if isinstance(benchmark, YAHPOBenchmark):
        pytest.skip(
            "YAHPOBench gives slight numerical instability when querying in bulk vs"
            " each config individually.",
        )

    query_results = [benchmark.query(config, at=f) for f in benchmark.iter_fidelities()]
    trajectory_results = benchmark.trajectory(config)

    for qr, tr in zip(query_results, trajectory_results):
        assert qr == tr, f"{qr}\n{tr}"


def test_trajectory_is_over_full_range_by_default(benchmark: Benchmark) -> None:
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


def test_argmin_query(benchmark: Benchmark) -> None:
    # Get a random configuration
    random_config = benchmark.sample()

    # Get the argmax
    argmin_config = benchmark.query(random_config, argmin="error")

    # Get the trajectory
    trajectory = benchmark.trajectory(random_config)
    best_in_trajectory = min(trajectory, key=lambda x: x.error)

    assert argmin_config == best_in_trajectory


def test_config_with_same_content_hashes_correctly(benchmark: Benchmark) -> None:
    config = benchmark.sample()

    if isinstance(benchmark, TabularBenchmark):
        config_dict = config.dict(with_id=True)
    else:
        config_dict = config.dict()

    # Turn it into a dict and back again
    new_config = benchmark.Config.from_dict(config_dict)

    assert hash(config) == hash(new_config)


def test_result_with_same_content_hashes_correctly(benchmark: Benchmark) -> None:
    config = benchmark.sample()
    result = benchmark.query(config)

    # Turn it into a dict and back again
    new_result = benchmark.Result.from_dict(
        config=config,
        fidelity=result.fidelity,
        result=result.dict(),
    )

    assert hash(result) == hash(new_result)


def test_result_same_value_but_different_fidelity_has_different_hash(
    benchmark: Benchmark,
) -> None:
    config = benchmark.sample()
    result = benchmark.query(config)

    # Turn it into a dict and back again
    new_result = benchmark.Result.from_dict(
        config=config,
        fidelity=result.fidelity - 1,
        result=result.dict(),
    )

    assert hash(result) != hash(new_result)


@parametrize_with_cases("item", cases=".", has_tag=~ft.has_tag("generic_tabular"))
def test_prior_from_yaml_file(item: BenchmarkTest, tmp_path: Path) -> None:
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


@parametrize_with_cases("item", cases=".", has_tag=~ft.has_tag("generic_tabular"))
def test_prior_from_json_file(item: BenchmarkTest, tmp_path: Path) -> None:
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


@parametrize_with_cases("item", cases=".", has_tag=~ft.has_tag("generic_tabular"))
def test_prior_from_config(item: BenchmarkTest) -> None:
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


@parametrize_with_cases("item", cases=".", has_tag=~ft.has_tag("generic_tabular"))
def test_prior_from_dict(item: BenchmarkTest) -> None:
    params = item.unpack()
    bench = mfpbench.get(**params)

    # Get a random config
    random_config = bench.sample()
    # Use the path of the saved config as the prior config
    prior_config = random_config.dict()

    params["prior"] = prior_config

    if isinstance(bench, TabularBenchmark):
        params["prior"] = {"id": prior_config.id}

    bench = mfpbench.get(**params)

    # The default configuration for the benchmark should be the same as the prior
    default = bench.space.get_default_configuration()
    assert default == random_config, f"{random_config}, {default}"
