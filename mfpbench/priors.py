from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import mfpbench
from mfpbench import Benchmark, MFHartmannBenchmark, YAHPOBenchmark
from mfpbench.util import pairs


def benchmarks(
    seed: int,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    conditional_spaces: bool = False,
) -> Iterator[Benchmark]:
    for name, cls in mfpbench._mapping.items():
        if only and not any(o in name for o in only):
            continue

        if exclude and any(e in name for e in exclude):
            continue

        if cls.has_conditionals and not conditional_spaces:
            continue

        if issubclass(cls, YAHPOBenchmark) and cls.instances is not None:
            for task_id in cls.instances:
                yield mfpbench.get(name=name, task_id=task_id, seed=seed)
        else:
            yield mfpbench.get(name=name, seed=seed)


def generate_priors(
    seed: int,
    nsamples: int,
    to: Path,
    quantiles: Iterable[tuple[str, float]],
    prefix: str | None = None,
    fidelity: int | float | None = None,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
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
        if fidelity:
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

        # Sort the results, putting the best at the back to match up with
        # the upper quantiles being better
        sorted_results = sorted(results, key=lambda r: r.error, reverse=True)

        # Sort quantiles by the value, so we can assert later that the actual
        # results are what we expect
        quantiles = sorted(quantiles, key=lambda q: q[1])

        quantile_results = [
            (name, sorted_results[int(nsamples * quantile)])
            for name, quantile in quantiles
        ]

        # Make sure the ordered quartile results make sense, i.e.
        # the result at quartile .1 should be worse than the one at .9
        if len(quantiles) > 1:
            for (_, better), (_, worse) in pairs(quantile_results):
                assert better.error >= worse.error

        name_components = []
        if prefix is not None:
            name_components.append(prefix)

        name_components.append(bench.basename)

        name = "-".join(name_components)

        path_priors = [
            (to / f"{name}-{prior_name}.yaml", result.config)
            for prior_name, result in quantile_results
        ]
        for path, prior in path_priors:
            prior.save(path)

        # For Hartmann, we need to do some extra work
        if isinstance(bench, MFHartmannBenchmark):
            optimum = bench.Config.from_dict(
                {f"X_{i}": x for i, x in enumerate(bench.Generator.optimum)}
            )

            # Reserved keyword
            assert not any("perfect" == q[0] for q in quantiles)
            path = to / f"{name}-perfect.yaml"
            optimum.save(path)
