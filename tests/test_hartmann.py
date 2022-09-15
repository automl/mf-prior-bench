from __future__ import annotations

from typing import Callable

from functools import partial
from itertools import chain, product

from pytest_cases import fixture, parametrize

from mfpbench.synthetic.hartmann import (
    MFHartmann3Benchmark,
    MFHartmann6Benchmark,
    MFHartmannBenchmark,
)

BENCH_SEED = 1337


@fixture
@parametrize(
    "cls, prior",
    chain(
        product([MFHartmann3Benchmark], MFHartmann3Benchmark.available_priors),
        product([MFHartmann6Benchmark], MFHartmann6Benchmark.available_priors),
    ),
)
def MFH(
    cls: type[MFHartmannBenchmark], prior: str
) -> Callable[..., MFHartmannBenchmark]:
    """Returns the class and the basic args"""
    return partial(cls, prior=prior, seed=BENCH_SEED)


def test_hartmann_priors_with_and_without_noise_added(
    MFH: Callable[..., MFHartmannBenchmark]
) -> None:
    """
    Expects
    -------
    * Both the benchmarks should have a prior
    * The benchmark with no noise added should remain as the selected prior
    * Each config value should be changed by noise
    * The prior of the config space should be updated accordingly
    """
    bench_no_noise = MFH()
    clean_prior = bench_no_noise.prior

    bench_with_noise = MFH(prior_noise_seed=1)
    noisy_prior = bench_with_noise.prior

    # Just validaty checks for mypy
    assert noisy_prior is not None
    assert clean_prior is not None
    assert bench_no_noise.available_priors is not None
    assert bench_no_noise._prior_arg is not None
    assert isinstance(bench_no_noise._prior_arg, str)

    # Check it's the same as the original one advertised
    original_prior = bench_no_noise.available_priors[bench_no_noise._prior_arg]
    assert clean_prior == original_prior

    # All values different
    for v1, v2 in zip(clean_prior.dict().values(), noisy_prior.dict().values()):
        assert v1 != v2

    # configspace seeded with these priors
    assert clean_prior == bench_no_noise.space.get_default_configuration()
    assert noisy_prior == bench_with_noise.space.get_default_configuration()


@parametrize("scale", [-5, -1, -0.125, 0.125, 1, 5])
@parametrize("seed", [1, 2, 3, 4, 5])
def test_hartmann_priors_noise_in_bounds(
    MFH: Callable[..., MFHartmann6Benchmark],
    scale: float,
    seed: int,
) -> None:
    """
    Expects
    -------
    * Should produce a valid config
    * These values should all be between 0 and 1
    """
    bench = MFH(prior_noise_seed=seed, prior_noise_scale=scale)

    config = bench.prior
    assert config is not None

    config.validate()
    assert all(0 <= x <= 1 for x in config.dict().values())


def test_hartmann_priors_noise_different_seeds_different_noise(
    MFH: Callable[..., MFHartmann6Benchmark],
) -> None:
    """
    Expects
    -------
    * Using different seeds should result in different noisy priors
    """
    bench1 = MFH(prior_noise_seed=1)
    bench2 = MFH(prior_noise_seed=2)

    assert bench1.prior != bench2.prior
