from __future__ import annotations

from typing import Iterator

import argparse
from pathlib import Path

import mfpbench
from mfpbench import Benchmark, MFHartmannBenchmark, YAHPOBenchmark

SEED = 1
N_SAMPLES = 10


def benchmarks(
    seed: int = SEED,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
) -> Iterator[Benchmark]:
    for name, cls in mfpbench._mapping.items():
        if only and not any(o in name for o in only):
            continue

        if exclude and any(e in name for e in exclude):
            continue

        if cls.has_conditionals:
            continue

        if issubclass(cls, YAHPOBenchmark) and cls.instances is not None:
            for task_id in cls.instances:
                yield mfpbench.get(name=name, task_id=task_id, seed=seed)
        else:
            yield mfpbench.get(name=name, seed=seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--nsamples", type=int, default=N_SAMPLES)
    parser.add_argument("--dataset", type=str, choices=list(mfpbench._mapping))
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--to", type=str, default="priors")
    parser.add_argument("--fidelity")
    parser.add_argument("--only", type=str, nargs="*")
    parser.add_argument("--exclude", type=str, nargs="*")
    args = parser.parse_args()

    for bench in benchmarks(seed=args.seed, only=args.only):
        at = args.fidelity if args.fidelity else bench.fidelity_range[1]

        configs = bench.sample(n=args.nsamples)
        results = [bench.query(config, at=at) for config in configs]

        sorted_configs = sorted(results, key=lambda r: r.error)

        good = sorted_configs[0]
        medium = sorted_configs[len(sorted_configs) // 2]
        bad = sorted_configs[-1]

        assert good.error <= medium.error <= bad.error

        pre = args.prefix

        if args.fidelity is not None:
            pre = f"at-{args.fidelity}"

        name_components = []
        if args.prefix is not None:
            name_components.append(args.prefix)

        if args.fidelity is not None:
            name_components.append(args.prefix)

        name = "-".join(name_components + [bench.basename])

        to = Path(args.to)
        to.mkdir(exist_ok=True)

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
