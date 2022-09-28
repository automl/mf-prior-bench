from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import mfpbench
from mfpbench import Benchmark

SEED = 1
N_SAMPLES = 25
EPSILON = 1e-3
MAX_ITERATIONS = 5_000

STYLES = {}


class RunningStats:
    # https://stackoverflow.com/a/17637351

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
        self.previous_m = 0
        self.previous_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1
        self.previous_m = self.old_m
        self.previous_s = self.old_s

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def std(self):
        return np.sqrt(self.variance())


def correlation_curve(
    b: Benchmark,
    *,
    n_samples: int = 25,
    method: str = "spearman",
) -> np.ndarray:
    configs = b.sample(n_samples)
    frame = b.frame()
    for config in configs:
        trajectory = b.trajectory(config)
        for r in trajectory:
            frame.add(r)

    correlations = frame.correlations(method=method)
    return correlations[-1, :]


def plot(
    stats: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    to: Path | None = None,
) -> None:
    if to is None:
        to = Path("correlations.png")

    print(stats)
    fig, ax = plt.subplots()

    for name, (mean, std) in stats.items():
        xs = np.linspace(0, 1, len(mean))
        style = STYLES.get(name, {})
        ax.plot(xs, mean, label=name, **style)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2)

    ax.set_xlim(auto=True)
    ax.set_ylim(auto=True)
    ax.set_ylabel("Spearman correlation")
    ax.set_xlabel("Fidelity %")
    ax.legend()

    print(f"Saving to {to}")
    plt.savefig(to)


def monte_carlo(
    benchmark: Benchmark,
    n_samples: int = 25,
    epsilon: float = 1e-3,
    iterations_max: int = 5000,
) -> RunningStats:
    stats = RunningStats()
    converged = False
    itrs = 0
    while not converged and itrs < iterations_max:
        curve = correlation_curve(b, n_samples=n_samples)
        stats.push(curve)

        if stats.n > 2:
            diff = np.linalg.norm(stats.new_m - stats.previous_m, ord=2)
            if diff <= epsilon:
                converged = True

        else:
            diff = np.inf
        print(itrs, diff)
        itrs += 1

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--benchmark", type=str)
    parser.add_argument("--task-id", type=str)
    parser.add_argument("--datadir", type=str, required=False)

    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--iterations_max", type=int, default=MAX_ITERATIONS)

    parser.add_argument("--results-dir", type=str, default="correlation_results")

    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--plot-only", nargs="*", required=False)
    parser.add_argument("--plot-to", type=str, default="test.pdf")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        results_dir.mkdir()

    if args.plot:
        only = args.plot_only

        if only is None:
            names = [
                f.stem
                for f in results_dir.iterdir()
                if f.is_file() and f.suffix == ".json"
            ]
        else:
            names = only

        results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in names:
            result_path = results_dir / f"{name}.json"
            with result_path.open("r") as f:
                result = json.load(f)
                results[name] = (np.array(result["mean"]), np.array(result["std"]))

        plot(results, to=args.plot_to)

    else:
        kwargs = dict(name=args.benchmark, seed=args.seed)

        if args.task_id:
            kwargs["task_id"] = args.task_id

        if args.datadir:
            datadir = Path(args.datadir)
            assert datadir.exists()
            kwargs["datadir"] = datadir

        b = mfpbench.get(**kwargs)
        stats = monte_carlo(
            benchmark=b,
            n_samples=args.n_samples,
            iterations_max=args.iterations_max,
            epsilon=args.epsilon,
        )

        results = {"mean": stats.mean().tolist(), "std": stats.std().tolist()}

        result_path = results_dir / f"{args.name}.json"
        with result_path.open("w") as f:
            json.dump(results, f)
