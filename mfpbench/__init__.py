from __future__ import annotations

from typing import Iterator

import datetime
from pathlib import Path

from mfpbench.benchmark import Benchmark
from mfpbench.jahs import (
    JAHSBenchmark,
    JAHSCifar10,
    JAHSColorectalHistology,
    JAHSFashionMNIST,
)
from mfpbench.yahpo import LCBenchBenchmark, YAHPOBenchmark

name = "mf-prior-bench"
package_name = "mfpbench"
author = "bobby1 and bobby2"
author_email = "eddiebergmanhs@gmail.com"
description = "No description given"
url = "https://www.automl.org"
project_urls = {
    "Documentation": "https://automl.github.io/mf-prior-bench/main",
    "Source Code": "https://github.com/automl/mfpbench",
}
copyright = f"Copyright {datetime.date.today().strftime('%Y')}, bobby1 and bobby2"
version = "0.0.1"

_mapping: dict[str, type[Benchmark]] = {
    "jahs_cifar_10": JAHSCifar10,
    "jahs_colorectal_histology": JAHSColorectalHistology,
    "jahs_fashion_mnist": JAHSFashionMNIST,
    "lcbench": LCBenchBenchmark,
}


def get(
    name: str,
    task_id: str | None = None,
    datadir: str | Path | None = None,
    seed: int | None = None,
) -> Benchmark:
    """Get a benchmark

    ```python
    import mfpbench
    b = mfpbench.get("jahs_cifar_10", datadir=..., seed=1)
    cs = b.space()

    sample = cs.sample_configuration()

    result = b.query(sample, at=42)
    print(result)
    ```

    If using something with a task:

    ```python
    b = mfpbench.get("lcbench", task="3945", datadir=..., seed=1)
    ```

    Parameters
    ----------
    name : str
        The name of the benchmark

    task_id : str | None = None
        A task id if it's an instance of a YAHPO benchmark, must be provided if a YAHPO
        benchmark

    datadir: str | Path | None = None
        Path to where the benchmark should look for data if it does. Defaults to
        "./data/<benchmark-specific>"

    seed: int | None = None
        The seed to use
    """
    b = _mapping.get(name, None)
    if b is None:
        raise ValueError(f"{name} is not a benchmark in {list(_mapping.keys())}")

    if issubclass(b, JAHSBenchmark):
        if task_id is not None:
            raise ValueError(f"jahs-bench doesn't take a task_id ({task_id})")

        return b(datadir=datadir, seed=seed)

    # TODO: this might have to change, not sure if all yahpo benchmarks have a task
    if issubclass(b, YAHPOBenchmark):
        return b(task_id=task_id, datadir=datadir, seed=seed)

    raise RuntimeError("Whoops, please fix me")


def available() -> Iterator[tuple[str, type[Benchmark], str | None]]:
    """Iterate over all the possible instantiations of a benchmark

    Returns
    -------
    Iterator[tuple[type[Benchmark], str | None]]
    """
    for k, v in _mapping.items():
        if issubclass(v, JAHSBenchmark):
            yield k, v, None
        elif issubclass(v, YAHPOBenchmark):
            if v.instances is not None:
                yield from ((k, v, task) for task in v.instances)
            else:
                yield (k, v, None)
    return


__all__ = [
    "name",
    "package_name",
    "author",
    "author_email",
    "description",
    "url",
    "project_urls",
    "copyright",
    "version",
    "JAHSCifar10",
    "JAHSColorectalHistology",
    "JAHSFashionMNIST",
    "JAHSConfigspace",
    "JAHSConfig",
    "JAHSResult",
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", action="store_true")
    args = parser.parse_args()

    if args.benchmarks is True:
        for name, cls, task in available():
            print(f"get(name={name}, task_id={task}) -> {cls.__name__}")
