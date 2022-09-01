import json
import pandas as pd
from pathlib import Path


class PD1Bench:
    def __init__(self, path: str, phase: int=1, matched: bool=False, load_on_init=True):
        self.base_path = Path(path)
        match_str = "matched" if matched else "unmatched"
        self.results_file_name = f"pd1_{match_str}_phase{phase}_results.jsonl"
        self.file_path = self.base_path / Path(self.results_file_name)

        self.results_df = None
        if load_on_init:
            self.load_benchmark()
        self.benchmark = None

    def load_benchmark(self):
        with open(self.file_path) as f:
            self.results_df = pd.read_json(f, orient="records", lines=True)
        return self.results_df

    def init_benchmark_task(self, dataset: str, batch_size: int):
        assert dataset in [
            'translate_wmt', 'uniref50', 'lm1b', 'svhn_no_extra', 'imagenet', 'mnist',
            'fashion_mnist', 'cifar100', 'cifar10'
        ]
        assert batch_size in [64, 128, 2048, 1024, 256, 512]

        _df = self.results_df.loc[self.results_df.dataset == dataset]
        _df = _df.loc[_df["hps.batch_size"] == batch_size]
        if len(_df) == 0:
            assert f"No tasks exist for batch_size {batch_size} and dataset {dataset}!"
        self.benchmark = _df.copy()

    def objective_function(self, config, fidelity):
        assert self.benchmark is not None, \
            "init_benchmark_task() needs to be called to create the benchmark task!"

        # TODO: locate the exact configuration and return loss, cost, etc. at an epoch (fidelity)
        pass