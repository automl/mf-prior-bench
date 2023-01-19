from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from tqdm import trange

import mfpbench
from mfpbench import Benchmark, MFHartmannBenchmark, YAHPOBenchmark
from mfpbench.result import Result


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

        kwargs = {
            "name": benchmark_name,
            "seed": seed,
        }
        if task_id is not None:
            kwargs["task_id"] = task_id

        benchmark = mfpbench.get(**kwargs)  # type: ignore

        yield benchmark


def generate_priors(
    seed: int,
    nsamples: int,
    to: Path,
    prior_spec: Iterable[tuple[str, int, float | None, float | None]],
    prefix: str | None = None,
    fidelity: int | float | None = None,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    use_hartmann_optimum: str | None = None,
    clean: bool = False,
) -> None:
    """Generate priors for a benchmark."""
    if to.exists() and clean:
        for child in filter(lambda path: path.is_file(), to.iterdir()):
            child.unlink()

    to.mkdir(exist_ok=True)
    prior_spec = list(prior_spec)

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

        results: list[Result] = []
        for _ in trange(nsamples, desc=f"Evaluating {bench.basename}"):
            config = bench.sample()
            result = bench.query(config, at=at)
            results.append(result)

        print(" - Finished results")  # noqa: T201
        results = sorted(results, key=lambda r: r.error)

        # Take out the results as specified by the prior and store the perturbations
        # to make, if any.
        prior_configs = {
            name: (results[index].config, std, categorical_swap_chance)
            for name, index, std, categorical_swap_chance in prior_spec
        }
        print(" - Priors: ", prior_configs)  # noqa: T201

        # Inject hartmann optimum in if specified
        if use_hartmann_optimum is not None and isinstance(bench, MFHartmannBenchmark):
            if use_hartmann_optimum not in prior_configs:
                raise ValueError(f"Prior '{use_hartmann_optimum}' not found in priors.")

            prior_configs[use_hartmann_optimum] = bench.optimum

        # Perturb each of the configs as specified to make the offset priors
        space = bench.space
        priors = {
            name: config.perturb(
                space,
                seed=seed,
                std=std,
                categorical_swap_chance=categorical_swap_chance,
            )
            for name, (config, std, categorical_swap_chance) in prior_configs.items()
        }
        print(" - Perturbed priors: ", priors)  # noqa: T201

        name_components = []
        if prefix is not None:
            name_components.append(prefix)

        name_components.append(bench.basename)

        name = "-".join(name_components)

        path_priors = [
            (to / f"{name}-{prior_name}.yaml", prior_config)
            for prior_name, prior_config in priors.items()
        ]
        for path, prior in path_priors:
            prior.save(path)
