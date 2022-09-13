from __future__ import annotations

from typing import Any, Iterator

import datetime

from mfpbench.benchmark import Benchmark
from mfpbench.jahs import (
    JAHSBenchmark,
    JAHSCifar10,
    JAHSColorectalHistology,
    JAHSFashionMNIST,
)
from mfpbench.synthetic.hartmann import (
    MFHartmann3Benchmark,
    MFHartmann3BenchmarkBad,
    MFHartmann3BenchmarkGood,
    MFHartmann3BenchmarkModerate,
    MFHartmann3BenchmarkTerrible,
    MFHartmann6Benchmark,
    MFHartmann6BenchmarkBad,
    MFHartmann6BenchmarkGood,
    MFHartmann6BenchmarkModerate,
    MFHartmann6BenchmarkTerrible,
    MFHartmannBenchmark,
)
from mfpbench.yahpo import (
    IAMLglmnetBenchmark,
    IAMLrangerBenchmark,
    IAMLrpartBenchmark,
    IAMLSuperBenchmark,
    IAMLxgboostBenchmark,
    LCBenchBenchmark,
    NB301Benchmark,
    RBV2aknnBenchmark,
    RBV2glmnetBenchmark,
    RBV2rangerBenchmark,
    RBV2rpartBenchmark,
    RBV2SuperBenchmark,
    RBV2svmBenchmark,
    RBV2xgboostBenchmark,
    YAHPOBenchmark,
)

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
    "jahs_cifar10": JAHSCifar10,
    "jahs_cifar10_good": JAHSCifar10,
    "jahs_cifar10_bad": JAHSCifar10,
    "jahs_colorectal_histology": JAHSColorectalHistology,
    "jahs_fashion_mnist": JAHSFashionMNIST,
    "lcbench": LCBenchBenchmark,
    "nb301": NB301Benchmark,
    "mfh3": MFHartmann3Benchmark,
    "mfh3_terrible": MFHartmann3BenchmarkTerrible,
    "mfh3_terrible_good": MFHartmann3BenchmarkTerrible,
    "mfh3_terrible_bad": MFHartmann3BenchmarkTerrible,
    "mfh3_bad": MFHartmann3BenchmarkBad,
    "mfh3_moderate": MFHartmann3BenchmarkModerate,
    "mfh3_good": MFHartmann3BenchmarkGood,
    "mfh3_good_good": MFHartmann3BenchmarkGood,
    "mfh3_good_bad": MFHartmann3BenchmarkGood,
    "mfh6": MFHartmann6Benchmark,
    "mfh6_terrible": MFHartmann6BenchmarkTerrible,
    "mfh6_terrible_good": MFHartmann6BenchmarkTerrible,
    "mfh6_terrible_bad": MFHartmann6BenchmarkTerrible,
    "mfh6_bad": MFHartmann6BenchmarkBad,
    "mfh6_moderate": MFHartmann6BenchmarkModerate,
    "mfh6_good": MFHartmann6BenchmarkGood,
    "mfh6_good_good": MFHartmann6BenchmarkGood,
    "mfh6_good_bad": MFHartmann6BenchmarkGood,
    "rbv2_super": RBV2SuperBenchmark,
    "rbv2_aknn": RBV2aknnBenchmark,
    "rbv2_glmnet": RBV2glmnetBenchmark,
    "rbv2_ranger": RBV2rangerBenchmark,
    "rbv2_rpart": RBV2rpartBenchmark,
    "rbv2_svm": RBV2svmBenchmark,
    "rbv2_xgboost": RBV2xgboostBenchmark,
    "iaml_glmnet": IAMLglmnetBenchmark,
    "iaml_ranger": IAMLrangerBenchmark,
    "iaml_rpart": IAMLrpartBenchmark,
    "iaml_super": IAMLSuperBenchmark,
    "iaml_xgboost": IAMLxgboostBenchmark,
}


def get(name: str, seed: int | None = None, **kwargs: Any) -> Benchmark:
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

    seed: int | None = None
        The seed to use

    **kwargs: Any
        Extra arguments, optional or required for other benchmarks

        YAHPOBenchmark
        --------------
        datadir: str | Path | None = None
            Path to where the benchmark should look for data if it does. Defaults to
            "./data/yahpo-gym-data"

        task_id : str | None = None
            A task id if the yahpo bench requires one

        JAHSBenchmark
        -------------
        datadir: str | Path | None = None
            Path to where the benchmark should look for data if it does. Defaults to
            "./data/jahs-bench-data"

        MFHartmann3Benchmark
        --------------------
        bias: float
            Amount of bias, realized as a flattening of the objective.

        noise: float
            Amount of noise, decreasing linearly (in st.dev.) with fidelity.
    """
    b = _mapping.get(name, None)
    if b is None:
        raise ValueError(f"{name} is not a benchmark in {list(_mapping.keys())}")

    if issubclass(b, JAHSBenchmark):
        if kwargs.get("task_id") is not None:
            raise ValueError(f"jahs-bench doesn't take a task_id ({kwargs['task_id']})")

        return b(
            seed=seed,
            datadir=kwargs.get("datadir"),
            prior=None if "prior" not in kwargs else kwargs["prior"],
        )

    # TODO: this might have to change, not sure if all yahpo benchmarks have a task
    if issubclass(b, YAHPOBenchmark):
        return b(
            seed=seed,
            datadir=kwargs.get("datadir"),
            task_id=kwargs.get("task_id"),
        )

    if issubclass(b, MFHartmannBenchmark):
        return b(
            seed=seed,
            bias=kwargs.get("bias"),
            noise=kwargs.get("noise"),
            prior=None if "prior" not in kwargs else kwargs["prior"],
        )

    raise RuntimeError("Whoops, please fix me")


def available(
    conditionals: bool = False,
) -> Iterator[tuple[str, type[Benchmark], dict[str, Any] | None]]:
    """Iterate over all the possible instantiations of a benchmark

    Parameters
    ----------
    conditionals: bool = False
        Whether to iterate through benchmarks with conditional hyperparameters.

    Returns
    -------
    Iterator[tuple[type[Benchmark], str | None]]
    """
    for k, v in _mapping.items():
        # Skip conditionals if specified
        if v.has_conditionals and conditionals is False:
            continue

        if issubclass(v, JAHSBenchmark):
            yield k, v, None

        elif issubclass(v, YAHPOBenchmark):
            if v.instances is not None:
                yield from ((k, v, {"task_id": task}) for task in v.instances)
            else:
                yield (k, v, None)

        elif issubclass(v, MFHartmannBenchmark):
            bias, noise = v.bias_noise
            extra = {"bias": bias, "noise": noise}
            yield (k, v, extra)

        else:
            raise NotImplementedError("Whoops, fix me")
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
    "available",
    "get",
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", action="store_true")
    parser.add_argument("--has-conditional-hps", action="store_true")
    args = parser.parse_args()

    if args.benchmarks is True:
        for name, cls, extra in available(conditionals=args.has_conditional_hps):
            if extra is None:
                print(f"get(name={name}) -> {cls.__name__}")
            elif issubclass(cls, MFHartmannBenchmark) and any(
                s in name for s in ["terrible", "good", "moderate", "bad"]
            ):
                print(f"get(name={name}) -> {cls.__name__}")
            else:
                kv_str = ", ".join([f"{k}={v}" for k, v in extra.items()])
                print(f"get(name={name}, {kv_str}) -> {cls.__name__}")
