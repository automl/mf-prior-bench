from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

import mfpbench
from mfpbench import Benchmark, MFHartmannBenchmark, YAHPOBenchmark
from mfpbench.result import Result
from mfpbench.util import pairs


def benchmarks(
    seed: int,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    conditional_spaces: bool = False,  # Note supported due to `remove_hyperparamter`
) -> Iterator[Benchmark]:

    # A mapping from the indexable name to the argument name and cls
    benches: dict[str, tuple[str, type[Benchmark], str | None]] = {}

    for name, cls in mfpbench._mapping.items():
        if issubclass(cls, YAHPOBenchmark) and cls.instances is not None:
            benches.update(
                {f"{name}-{task_id}": (name, cls, task_id) for task_id in cls.instances}
            )
        else:
            benches[name] = (name, cls, None)

    for index_name, (benchmark_name, cls, task_id) in benches.items():
        if only is not None and not any(index_name.startswith(o) for o in only):
            continue

        if exclude is not None and any(index_name.startswith(e) for e in exclude):
            continue

        if cls.has_conditionals and not conditional_spaces:
            continue

        if task_id is not None:
            benchmark = mfpbench.get(name=benchmark_name, task_id=task_id, seed=seed)
        else:
            benchmark = mfpbench.get(name=benchmark_name, seed=seed)

        yield benchmark


def generate_priors(
    seed: int,
    nsamples: int,
    to: Path,
    quantiles: Iterable[tuple[str, float]],
    prefix: str | None = None,
    fidelity: int | float | None = None,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    hartmann_perfect: bool = True,
    hartmann_optimum_with_noise: Iterable[tuple[str, float]] | None = None,
    clean: bool = False,
) -> None:
    """Generate priors for a benchmark."""
    if to.exists() and clean:
        for child in filter(lambda path: path.is_file(), to.iterdir()):
            child.unlink()

    to.mkdir(exist_ok=True)

    for bench in benchmarks(seed=seed, only=only, exclude=exclude):

        max_fidelity = bench.fidelity_range[1]

        # If a fidelity was specfied, then we need to make sure we can use it
        # as an int in a benchmark with an int fidelity, no accidental rounding.
        if fidelity is not None:
            if isinstance(max_fidelity, int) and isinstance(fidelity, float):
                if fidelity.is_integer():
                    fidelity = int(fidelity)

            if type(fidelity) != type(max_fidelity):
                raise ValueError(
                    f"Cannot use fidelity {fidelity} (type={type(fidelity)}) with"
                    f" benchmark {bench.basename}"
                )
            at = fidelity
        else:
            at = max_fidelity

        configs = bench.sample(n=nsamples)
        results = [bench.query(config, at=at) for config in configs]
        results = sorted(results, key=lambda result: result.error)

        errors = np.asarray([result.error for result in results])

        def get_result(q: float, _errors: np.ndarray, _results: list[Result]) -> Result:
            quantile_value = np.quantile(_errors, q)
            indices_below_quantile = np.argwhere(_errors <= quantile_value).flatten()
            selected = indices_below_quantile[-1]
            return _results[selected]

        quantile_results = [
            (name, q, get_result(q, errors, results)) for name, q in quantiles
        ]

        # Sort quantiles by the value, so we can assert later that the actual
        # results are what we expect
        # Make sure the ordered quartile results make sense, i.e.
        # the result at quartile .1 should be worse than the one at .9
        quantile_results = sorted(quantile_results, key=lambda q: q[1])
        if len(quantile_results) > 1:
            for (_, _, better), (_, _, worse) in pairs(quantile_results):
                assert better.error <= worse.error

        name_components = []
        if prefix is not None:
            name_components.append(prefix)

        name_components.append(bench.basename)

        name = "-".join(name_components)

        path_priors = [
            (to / f"{name}-{prior_name}.yaml", result.config)
            for prior_name, _, result in quantile_results
        ]
        for path, prior in path_priors:
            prior.save(path)

        # For Hartmann, we need to do some extra work for optimum
        if hartmann_perfect and isinstance(bench, MFHartmannBenchmark):
            # Reserved keyword
            assert not any(qname == "perfect" for qname, _, _ in quantile_results)
            optimum = bench.Config.from_dict(
                {f"X_{i}": x for i, x in enumerate(bench.Generator.optimum)}
            )
            # Generate a perfect prior, the prior which is located at the optimum
            optimum.save(to / f"{name}-perfect.yaml")

        if hartmann_optimum_with_noise and isinstance(bench, MFHartmannBenchmark):
            hartmann_optimum_with_noise = list(hartmann_optimum_with_noise)

            # Reserved keyword
            assert not any(
                pname == "perfect" for pname, _ in hartmann_optimum_with_noise
            ), "Reserved keyword"

            optimum = bench.Config.from_dict(
                {f"X_{i}": x for i, x in enumerate(bench.Generator.optimum)}
            )

            for prior, noise in hartmann_optimum_with_noise:
                assert prior != "perfect", "reserved keyword"

                rng = np.random.default_rng(seed=seed)
                uniform = rng.uniform(low=-1, high=1, size=bench.dims)
                noise_values = uniform * noise

                config = optimum.from_dict(
                    {
                        k: float(np.clip(v + n, a_min=0, a_max=1))
                        for (k, v), n in zip(optimum.items(), noise_values)
                    }
                )
                config.save(to / f"{name}-{prior}.yaml")
