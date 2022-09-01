from __future__ import annotations

from abc import ABC
from typing import Any

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace

from mfpbench.benchmark import Benchmark
from mfpbench.download import DATAROOT
from mfpbench.pd1.config import PD1Config
from mfpbench.pd1.result import PD1Result
from mfpbench.pd1.space import pd1_configspace


class PD1Benchmark(Benchmark, ABC):
    """Manages access to the PD1 tabular data"""

    Config = PD1Config
    Result = PD1Result

    # TODO: Need to clarify this
    fidelity_name = "epoch"

    # TODO: Have to figure this out per dataset probably
    # if they differ per dataset, then create a new class PD1BenchmarkDataset
    # with the class variable set to `fidelity_range = (start, end, step)`
    fidelity_range = (1, 200, 1)

    _tasks = (
        "translate_wmt",
        "uniref50",
        "lm1b",
        "svhn_no_extra",
        "imagenet",
        "mnist",
        "fashion_mnist",
        "cifar100",
        "cifar10",
    )

    # Where the data for jahsbench should be located relative to the path given
    _default_download_dir: Path = DATAROOT / "pd1-data"

    def __init__(
        self,
        *,
        task: str,
        full: bool = True,
        matched: bool = True,
        datadir: str | Path | None = None,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        task: str
            A task in "translate_wmt", "uniref50", "lm1b", "svhn_no_extra", "imagenet",
            "mnist", "fashion_mnist", "cifar100", "cifar10",

        full: bool = True
            Whether to use the preliminary data collection set or the scaled up version
            which we dub "full"

        matched: bool = True
            Whether the configs for the datasets are the same across each dataset.
            For consistency and simplicity, we only accept true for this parameter but
            leave it here as a note

        datadir : str | Path | None = None
            The path to where mfpbench stores it data. If left to default (None), will
            use the `_default_download_dir = ./data/jahs-bench-data`.

        seed : int | None = None
            The seed to give this benchmark instance

        """
        super().__init__(seed=seed)
        assert matched, "Only accepts `matched = True` for now for tabular dataset"

        if not full:
            raise NotImplementedError(
                "Likely needs to consider seperate space for batch size as indicated"
                " in README."
            )

        if datadir is None:
            datadir = PD1Benchmark._default_download_dir

        self.datadir = Path(datadir) if isinstance(datadir, str) else datadir
        if not self.datadir.exists():
            raise FileNotFoundError(
                f"Can't find folder at {self.datadir}, have you run\n"
                f"`python -m mfpbench.download --data-dir {self.datadir.parent}`"
            )

        self.full = full
        self.matched = matched
        self.task = task
        self._configspace = pd1_configspace(self.seed, matched=matched, full=full)
        self._df: pd.DataFrame | None = None

    # explicit overwrite
    def load(self) -> None:
        """Pre-load PD1 XGBoost model before querying the first time"""
        self.df  # Access it to make it load

    @property
    def space(self) -> ConfigurationSpace:
        """The ConfigurationSpace for this benchmark"""
        return self._configspace

    def query(
        self,
        config: PD1Config | dict[str, Any] | Configuration,
        at: int | None = None,
    ) -> PD1Result:
        """Query the results for a config

        Parameters
        ----------
        config : PD1Config | dict[str, Any] | Configuration
            The config to query

        at : int | None = None
            The epoch at which to query at, defaults to max (200) if left as None

        Returns
        -------
        PD1Result
            The result for the config at the given epoch
        """
        at = at if at is not None else self.end
        assert self.start <= at <= self.end

        if isinstance(config, Configuration):
            config = self.Config.from_dict({**config})

        if isinstance(config, dict):
            config = self.Config.from_dict(config)

        assert isinstance(config, self.Config), f"Nope, config is {type(config)}"

        return self._results_for(config, fidelities=[at])[0]

    def trajectory(
        self,
        config: PD1Config | dict[str, Any] | Configuration,
        *,
        frm: int | None = None,
        to: int | None = None,
        step: int | None = None,
    ) -> list[PD1Result]:
        """Query the trajectory of a config as it ranges over a fidelity

        Parameters
        ----------
        config : PD1Config | dict[str, Any] | Configuration
            The config to query

        frm: int | None = None
            Start of the curve, defaults to the minimum fidelity (1)

        to: int | None = None
            End of the curve, defaults to the maximum fidelity (200)

        step: int | None = None
            Step size, defaults to benchmark standard (1 for epoch)

        Returns
        -------
        list[PD1Result]
            The results over that trajectory
        """
        if isinstance(config, Configuration):
            config = self.Config.from_dict({**config})

        if isinstance(config, dict):
            config = self.Config.from_dict(config)

        assert isinstance(config, self.Config), f"Nope, config is {type(config)}"

        fidelities = list(self.iter_fidelities(frm, to, step))

        return self._results_for(config, fidelities=fidelities)

    @property
    def df(self) -> pd.DataFrame:
        if not self._df:
            m = "matched" if self.matched else "unmatched"
            p = "phase1" if self.full else "phase0"

            # TODO: create these in the download and process step
            # Could optionally split out the big
            # The columns should be like the following, (order doesnt matter)
            #
            #   epoch | hp_1 | hp_2 | .... | result_1 | result_2 | ...
            #
            # These `hp_1`, ... `hp_n` should match with the attributes on the
            # corresponding PD1Config.
            #
            # These `result_1`, ... `result_n` should match with the attributes on the
            # corresponding PD1Result.
            path = self.datadir / f"pd1_{m}_{p}_{self.task}_results.csv"
            raise NotImplementedError()

            self._df = pd.read_csv(path)

        return self._df

    def _raw_data(self) -> pd.DataFrame:
        """Get the raw data which is not so helpful, holds info for other datasets..."""
        m = "matched" if self.matched else "unmatched"
        p = "phase1" if self.full else "phase0"
        path = self.datadir / f"pd1_{m}_{p}_results.jsonls"
        with open(path, "r") as f:
            return pd.read_json(f, orient="records", lines=True)

    def _results_for(self, config: PD1Config, fidelities: list[int]) -> list[PD1Result]:
        # Here we expect each entry in the config matches a colum
        c = asdict(config)
        print(config)

        # hps = [hp1, hp2], values = [a, a], fidelities = [1, 2]
        hps = list(c.keys())
        values = list(c.values())

        #   index | epoch | hp1 | hp2 | result1 | result2
        # >   .       1      a     a  |   xx    |   xx
        # >   .       2      a     a  |   xx    |   xx
        #     .       3      a     a  |   xx    |   xx
        #     .       1      b     b  |   xx    |   xx
        #     .       2      b     b  |   xx    |   xx
        #     .       3      b     b  |   xx    |   xx
        df = self.df

        # Grab the rows where the config matches
        #
        #   index | epoch | hp1 | hp2 | result1 | result2
        # >   .       1      a     a  |   xx    |   xx
        # >   .       2      a     a  |   xx    |   xx
        #     .       3      a     a  |   xx    |   xx
        #
        #
        matching_rows = (df[hps] == values).all(axis=1)
        selected = df[matching_rows]

        # Drop the hp colums
        #
        #   index | epoch | result1 | result2
        # >   .       1   |   xx    |   xx
        # >   .       2   |   xx    |   xx
        #     .       3   |   xx    |   xx
        selected = selected.drop(columns=hps)

        # Grab the rows where the fidelity matches
        #   index | epoch | result1 | result2
        # >   .       1   |   xx    |   xx
        # >   .       2   |   xx    |   xx
        rows = selected[selected[self.fidelity_name].isin(fidelities)]

        assert len(rows) == len(fidelities), "Should have one per fidelity"

        # We now have unique fidelities for this config, lets make it the index and sort
        #
        #   epoch (index) | result1 | result2 | ...
        # > 1                 xx       xx
        # > 2                 xx       xx
        #
        results = rows.set_index(self.fidelity_name)
        results = results.sort_index()

        # Now we create a seperate dictionary for each
        # {
        #   1: { "result1": xx, "result2": xx },
        #   2: { "result1": xx, "result2": xx },
        # }
        results_by_fidelity: dict[int, dict[str, Any]] = results.to_dict(orient="index")

        # TODO: Need to implement PD1Result to match the results present
        print(list(results_by_fidelity[fidelities[0]].keys()))

        # Finally, we create our results and return
        return [
            self.Result.from_dict(
                config=config,
                fidelity=f,
                result=results_by_fidelity[f],
            )
            for f in fidelities
        ]


class PD1BenchmarkExample(PD1Benchmark):
    """Just an example"""

    fidelity_range = (1, 199, 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()

    task = PD1Benchmark._tasks[0]
    bench = PD1Benchmark(seed=1, datadir=args.datadir, task=task)

    # Print out the contents of the data dir
    print(bench.datadir.iterdir())

    # Print what you want here
    raw_data = bench._raw_data()

    # ... that hard part here is basically figuring out how to make a nice and easy to
    # use dataframe with the info we need in it.
    # * What to use as hyperparameters
    # * How to configspace it
    # * Split up this raw data into being dataset specific

    sample = bench.sample()
    results = bench.trajectory(sample)

    print(results)
