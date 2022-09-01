from __future__ import annotations

from typing import cast

import numpy as np

import mfpbench

# Optional imports here just to give typing and so you can look deeper if needed
from mfpbench.jahs import JAHSBenchmark, JAHSConfig, JAHSResult

if __name__ == "__main__":

    seed = 724

    # Just adding a type here if you want to explore it
    benchmark = mfpbench.get("jahs_cifar10", seed=seed)  # datadir = ...
    benchmark = cast(JAHSBenchmark, benchmark)

    # benchmark = mfpbench.get("lcbench", seed=seed, datadir=datadir, task_id="3945")
    # benchmark = cast(LCBench, benchmark)

    # Get a random config just to see it
    config: JAHSConfig = benchmark.sample()

    # And the search space
    print(benchmark.space)

    print("\n")
    print(config)  # It's a dataclass
    print(config.dict())  # Cause you might want this
    print("\n")

    # Copy a configuration if you need to for any reason, these are two distinct objects
    exact_copy = config.copy()
    assert exact_copy == config

    # You can only mutate it by a copy to keep things consistent
    new_copy = config.copy(TrivialAugment=True)

    # You can always validate a config to make sure you aren't doing anything wrong
    new_copy.validate()

    # Like in this case where we used a bad optmizer
    bad_copy = config.copy(Optimizer="Adam")
    try:
        bad_copy.validate()
    except AssertionError:
        print("\n")
        print("Error was raised while validating")
        print("\n")

    # Anyways, here's the results for the config
    result: JAHSResult = benchmark.query(config, at=42)
    result = benchmark.query(config.dict(), at=42)  # You can also use a dict
    result = benchmark.query(benchmark.space.sample_configuration())  # Or configspace

    # All results have the following properties mapped to some metrics that are returned
    # by the benchmark
    print(result.score)
    print(result.error)
    print(result.cost)

    print(result.test_error)
    print(result.test_score)
    print(result.val_error)
    print(result.val_score)

    # The full result object
    print(result)

    # And if you need the full trajectory
    results: list[JAHSResult] = benchmark.trajectory(config)
    sliced_result = benchmark.trajectory(config, to=100)
    sliced_result_2 = benchmark.trajectory(config, frm=50, to=100)

    first = results[0]
    last = results[-1]

    # Lets sort by score (each result type lists the score, "test-acc" for JAHS)
    sorted_result = sorted(results, key=lambda r: r.score, reverse=True)
    best = sorted_result[0]

    print("\n")
    print(f"first score = {first.score:.4f} | fidelity = {first.fidelity}")
    print(f"last score = {last.score:.4f} | fidelity = {last.fidelity}")
    print(f"best score = {last.score:.4f} | fidelity = {best.fidelity}")
    print("\n")

    # Now here's 100 configs and we'll get the best configuration
    configs: list[JAHSConfig] = benchmark.sample(100)

    # Get all trajectories for each run
    # [
    #   Run1:   [0, 1, 2, ..., 200]
    #           ...
    #   Run100: [0, 1, 2, ..., 200]
    # ]
    trajectories = [benchmark.trajectory(c) for c in configs]
    transposed = np.transpose(trajectories).tolist()  # type: ignore

    # {
    #    0:     [Run1, ..., Run100]
    #               ...
    #    200:   [Run1, ..., Run100]
    # }
    results_by_epoch = {
        epoch: results for epoch, results in enumerate(transposed, start=1)
    }

    # {
    #   0: RunX
    #   1: RunY
    #    ...
    #   200: RunZ
    # }
    best_per_epoch = {
        epoch: max(results, key=lambda r: r.score)
        for epoch, results in results_by_epoch.items()
    }

    for epoch in [1, 25, 50, 100]:
        print(f"epoch = {epoch}\n{best_per_epoch[epoch]}\n\n")
