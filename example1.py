from __future__ import annotations

import numpy as np

from mfpbench import JAHSCifar10, JAHSConfig, JAHSResult

# There's also JAHSFashionMNIST, JAHSColorectalHistology benchmarks
# They share the same config and results given back so it makes things easier
# YAHPO will need a specific trio of (benchmark, config, results) for each
# bench we consider in it

if __name__ == "__main__":
    seed = 724
    benchmark = JAHSCifar10()

    # Get a random config just to see it
    config: JAHSConfig
    config = benchmark.sample(seed=seed)

    print("\n")
    print(config)  # It's a dataclass
    print(config.dict())  # Cause you might want this
    print("\n")

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
    result: JAHSResult = benchmark.query(config, fidelity=42)

    # And if you need the full trajectory
    results: list[JAHSResult] = benchmark.trajectory(config)
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
    configs: list[JAHSConfig] = benchmark.sample(100, seed=seed)

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
