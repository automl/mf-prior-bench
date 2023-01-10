from __future__ import annotations

from pathlib import Path
from typing import Iterator

import mfpbench
from mfpbench import Benchmark, MFHartmannBenchmark, YAHPOBenchmark


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

        sorted_configs = sorted(results, key=lambda r: r.error)

        good = sorted_configs[0]
        medium = sorted_configs[len(sorted_configs) // 2]
        bad = sorted_configs[-1]

        assert good.error <= medium.error <= bad.error

        name_components = []
        if prefix is not None:
            name_components.append(prefix)

        name_components.append(bench.basename)

        name = "-".join(name_components)

        path_priors = [
            (to / f"{name}-good.yaml", good.config),
            (to / f"{name}-bad.yaml", bad.config),
            (to / f"{name}-medium.yaml", medium.config),
        ]
        for path, prior in path_priors:
            prior.save(path)

        # For Hartmann, we need to do some extra work
        if isinstance(bench, MFHartmannBenchmark):
            optimum = bench.Config.from_dict(
                {f"X_{i}": x for i, x in enumerate(bench.Generator.optimum)}
            )

            path = to / f"{name}-perfect.yaml"
            optimum.save(path)
