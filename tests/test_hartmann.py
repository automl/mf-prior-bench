from __future__ import annotations

from functools import partial
from itertools import chain, product
from typing import Callable

from mfpbench.synthetic.hartmann import (
    MFHartmann3Benchmark,
    MFHartmann6Benchmark,
    MFHartmannBenchmark,
)
from pytest_cases import fixture, parametrize

BENCH_SEED = 1337


@fixture
@parametrize(
    "cls, prior",
    chain(
        product([MFHartmann3Benchmark], ["good", "medium", "bad"]),
        product([MFHartmann6Benchmark], ["good", "medium", "bad"]),
    ),
)
@parametrize("seed", [1, 2, 3, 4, 5])
def MFH(
    cls: type[MFHartmannBenchmark], prior: str, seed: int
) -> Callable[..., MFHartmannBenchmark]:
    """Returns the class and the basic args."""
    return partial(cls, prior=prior, seed=seed)


def test_hartmann_priors_with_and_without_noise_added(
    MFH: Callable[..., MFHartmannBenchmark]
) -> None:
    """
    Expects
    -------
    * Both the benchmarks should have a prior
    * The benchmark with no noise added should remain as the selected prior
    * Each config value should be changed by noise
    * The prior of the config space should be updated accordingly.
    """
    bench_no_noise = MFH()
    clean_prior = bench_no_noise.prior

    bench_with_noise = MFH(noisy_prior=True)
    noisy_prior = bench_with_noise.prior

    # Just validaty checks for mypy
    assert noisy_prior is not None
    assert clean_prior is not None
    assert bench_no_noise._prior_arg is not None
    assert isinstance(bench_no_noise._prior_arg, str)

    # All values different
    for v1, v2 in zip(clean_prior.dict().values(), noisy_prior.dict().values()):
        assert v1 != v2

    # configspace seeded with these priors
    assert clean_prior == bench_no_noise.space.get_default_configuration()
    assert noisy_prior == bench_with_noise.space.get_default_configuration()


@parametrize("scale", [-5, -1, -0.125, 0.125, 1, 5])
def test_hartmann_priors_noise_in_bounds(
    MFH: Callable[..., MFHartmann6Benchmark],
    scale: float,
) -> None:
    """
    Expects
    -------
    * Should produce a valid config
    * These values should all be between 0 and 1.
    """
    bench = MFH(noisy_prior=True, prior_noise_scale=scale)

    config = bench.prior
    assert config is not None

    config.validate()
    for x in config.dict().values():
        assert 0 <= x <= 1


def test_hartmann_priors_noise_different_seeds_different_noise(
    MFH: Callable[..., MFHartmann6Benchmark],
) -> None:
    """
    Expects
    -------
    * Using different seeds should result in different noisy priors.
    """
    bench1 = MFH(noisy_prior=True, seed=1)
    bench2 = MFH(noisy_prior=True, seed=2)

    assert bench1.prior != bench2.prior
