from __future__ import annotations

from typing import cast

import numpy as np

import mfpbench

if __name__ == "__main__":
    seed = 724

    # Just adding a type here if you want to explore it # get(..., datadir = ...)
    benchmark: mfpbench.Benchmark = mfpbench.get("lcbench", task_id="3945", seed=seed)

    min_fidelity, max_fidelity, fidelity_step = benchmark.fidelity_range

    # benchmark = mfpbench.get("lcbench", seed=seed, datadir=datadir, task_id="3945")
    # benchmark = cast(LCBench, benchmark)

    # Get a random config just to see it
    config: mfpbench.Config = benchmark.sample()

    # And the search space

    # Copy a configuration if you need to for any reason, these are two distinct objects
    exact_copy = config.copy()
    assert exact_copy == config

    # You can only mutate it by a copy to keep things consistent
    new_copy = config.copy(momentum=0.2)

    # You can always validate a config to make sure you aren't doing anything wrong
    new_copy.validate()

    # Like in this case where we used a bad optmizer
    bad_copy = config.copy(momentum=-10)
    try:
        bad_copy.validate()
    except AssertionError:
        pass

    # Anyways, here's the results for the config
    result: mfpbench.Result
    result = benchmark.query(config, at=42)
    result = benchmark.query(config.dict(), at=42)  # You can also use a dict
    result = benchmark.query(benchmark.space.sample_configuration())  # Or configspace

    # All results have the following properties mapped to some metrics that are returned
    # by the benchmark

    # The full result object

    # And if you need the full trajectory
    results: list[mfpbench.Result] = benchmark.trajectory(config)
    sliced_result = benchmark.trajectory(config, to=max_fidelity)
    sliced_result_2 = benchmark.trajectory(
        config,
        frm=max_fidelity // 2,
        to=max_fidelity,
    )

    first = results[0]
    last = results[-1]

    # Lets sort by score (each result type lists the score, "test-acc" for JAHS)
    sorted_result = sorted(results, key=lambda r: r.score, reverse=True)
    best = sorted_result[0]

    # Now here's 100 configs and we'll get the best configuration
    configs: list[mfpbench.Config] = benchmark.sample(100)

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

    for _epoch in [1, 25, 50, 100]:
        pass
