from __future__ import annotations

from typing import Iterator

import gzip
import json
import shutil
from dataclasses import dataclass
from itertools import accumulate, product
from pathlib import Path

import numpy as np
import pandas as pd

from mfpbench.pd1.processing.columns import COLUMNS


def safe_accumulate(x: Iterator[float | None], fill: float = np.inf) -> Iterator[float]:
    itr = iter(f if f is not None else fill for f in x)
    return accumulate(itr)


def uniref50_epoch_convert(x: float | list[float]) -> float | list[float]:
    """
    Converts:
        0             NaN
        1    [0, 0, 0, 1]
        2           [nan]
        3     [0, 0, nan]

    to:
        0             NaN
        1    [0, 1, 2, 3]
        2           [nan]
        3     [0, 1, nan]
    """
    if isinstance(x, list):
        return [i if not pd.isna(e) else e for i, e in enumerate(x, start=1)]
    else:
        return x


@dataclass(frozen=True)
class Datapack:
    matched: bool
    phase: int
    dir: Path

    @property
    def rawname(self) -> str:
        m = "matched" if self.matched else "unmatched"
        return f"pd1_{m}_phase{self.phase}_results"

    @property
    def archive_path(self) -> Path:
        fname = self.rawname + ".jsonl.gz"
        return self.dir / fname

    def unpack(self) -> pd.DataFrame:
        frm = self.archive_path

        if not frm.exists():
            raise FileNotFoundError(f"No archive found at {frm}")

        with gzip.open(frm, mode="rt") as f:
            data = [json.loads(line) for line in f]

        return pd.DataFrame(data)


def process_pd1(tarball: Path, handle_nans: bool = False) -> None:
    datadir = tarball.parent
    rawdir = tarball.parent / "raw"

    fulltable_name = "full.csv"
    fulltable_path = datadir / fulltable_name

    # If we have the full table we can skip the tarball extraction
    if not fulltable_path.exists():
        if not tarball.exists():
            raise FileNotFoundError(f"No tarball found at {tarball}")

        rawdir.mkdir(exist_ok=True)

        # Unpack it to the rawdir
        print(f"Unpacking {tarball}")
        shutil.unpack_archive(tarball, rawdir)

        unpacked_folder_name = "pd1"  # This is what the tarball will unpack into
        unpacked_folder = rawdir / unpacked_folder_name

        # Move everything from the uncpack folder to the "raw" folder
        print(f"Moving files from {unpacked_folder} to {rawdir}")
        for filepath in unpacked_folder.iterdir():
            to = rawdir / filepath.name
            print(f"Move {filepath} to {to}")
            shutil.move(str(filepath), str(to))

        # Remove the archive folder, its all been moved to "raw"
        shutil.rmtree(str(unpacked_folder))

    # For processing the df
    drop_columns = [c.name for c in COLUMNS if not c.keep]
    renames = {c.name: c.rename for c in COLUMNS if c.rename is not None}
    hps = [c.rename for c in COLUMNS if c.hp]
    metrics = [c.rename if c.rename else c.name for c in COLUMNS if c.metric]

    dfs: list[pd.DataFrame] = []
    for matched, phase in product([True, False], [0, 1]):
        # Unpack the jsonl.gz archive if needed
        datapack = Datapack(matched=matched, phase=phase, dir=rawdir)
        print(f"Unpacking {datapack.archive_path}")
        df = datapack.unpack()

        # Tag them from the dataset they came from
        df["matched"] = matched
        df["phase"] = phase

        df.drop(columns=drop_columns, inplace=True)
        df.rename(columns=renames, inplace=True)

        dfs.append(df)

    # We now merge them all into one super table for convenience
    print("Creating master table for all datasets, phases and matches")
    full_df = pd.concat(dfs, ignore_index=True)

    # Since some columns values are essentially lists, we need to explode them out
    # However, we need to make sure to distuinguish between transformer and not as
    # transformers do not have test error available
    # We've already renamed the columns them at this point
    list_columns = [c.rename if c.rename else c.name for c in COLUMNS if c.type == list]
    transformer_datasets = ["uniref50", "translate_wmt", "imagenet", "lm1b"]
    dataset_columns = ["dataset", "model", "batch_size"]

    for (name, model, batchsize), dataset in full_df.groupby(dataset_columns):
        fname = f"{name}-{model}-{batchsize}"
        print(f"Exploding dataset {fname}")

        if name in transformer_datasets:
            explode_columns = [c for c in list_columns if c != "test_error_rate"]
            dataset.drop(columns=["test_error_rate"], inplace=True)
        else:
            explode_columns = list_columns

        if name == "uniref50":
            # For some reason the epochs of this datasets are basically [0, 0, 0, 1]
            # We just turn this into an incremental thing
            dataset["epoch"] = dataset["epoch"].apply(uniref50_epoch_convert)

        dataset["train_cost"] = [
            None if r is None else list(safe_accumulate(r, fill=np.inf))
            for r in dataset["train_cost"]
        ]

        dataset = dataset.explode(explode_columns, ignore_index=True)
        print(f"Writing individual dataset {fname}")

        if name == "lm1b":
            # Some train costs go obsenly high for no reason, we drop these rows
            dataset = dataset[dataset["train_cost"] < 10_000]
        elif name == "uniref50":
            # Some train costs go obsenly high for no reason, we drop these rows
            # Almost all are below 400 but we add a buffer
            dataset = dataset[dataset["train_cost"] < 4_000]

        # We want to write the full mixed {phase,matched} for surrogate training while
        # only keeping matched phase 1 data for tabular.
        # We also no longer need to keep dataset, model and batchsize for individual
        # datasets.
        # We can also drop "activate_fn" for all but 4 datasets
        has_activation_fn = [
            "fashion_mnist-max_pooling_cnn-256",
            "fashion_mnist-max_pooling_cnn-2048",
            "mnist-max_pooling_cnn-256",
            "mnist-max_pooling_cnn-2048",
        ]
        drop_columns = ["dataset", "model", "batch_size"]
        if fname not in has_activation_fn:
            drop_columns += ["activation_fn"]

        dataset = dataset.drop(columns=drop_columns)

        if handle_nans == "fill":
            raise NotImplementedError("TODO")
            # We now want it that for every epoch that is missing, we fill in partial
            # learning curves with
            #   train_cost -> worst in dataset
            #   metric -> worst possible score
            valid_epochs = dataset["epoch"].unique()
            valid_epochs = valid_epochs[~np.isnan(valid_epochs)]  # Remove nan

            hp_cols = list(hps)
            if name in transformer_datasets:
                hp_cols.remove("activation_fn")

            additional_data: list[pd.Series] = []

            config_groups = list(dataset.groupby(hps))

            # We find the worst training_cost entries for each row and just use those
            config_with_maximum_train_cost = max(
                [d for _, d in config_groups],
                lambda d: max(d["train_cost"]),
            )

            for hp_tuple, config_df in config_groups:

                epochs = config_df["epoch"]
                # Case 1: Has all valid_epochs, nothing to do
                if all(epochs == valid_epochs):
                    continue
                # Case 2: Has only a partial learning curve
                elif not any(epochs.isna()):
                    ...
                # Case 3: Has only a nan
                elif all(epochs.isna()):
                    ...
                    # train_fill = config_with_maximum_train_cost["train_cost"]
                else:
                    raise NotImplementedError(f"Didn't account for {epochs}")

        # Select only the tabular part (matched and phase1)
        """
        tabular_path = datadir / f"{fname}_tabular.csv"
        tabular_mask = dataset["matched"] & (dataset["phase"] == 1)
        df_tabular = dataset[tabular_mask]
        df_tabular = df_tabular.drop(columns=["matched", "phase"])

        print(f"Writing tabular data to {tabular_path}")
        df_tabular.to_csv(tabular_path, index=False)
        """

        # There are some entries which seem to appear twice. This is due to the same
        # config in {phase0,phase1} x {matched, unmatched}
        # To prevent issues, we simply drop duplicates
        hps = ["lr_decay_factor", "lr_initial", "lr_power", "opt_momentum"]
        if fname in has_activation_fn:
            hps += ["activation_fn"]

        dataset = dataset.drop_duplicates(hps + ["epoch"], keep="last")

        # The rest can be used for surrogate training
        surrogate_path = datadir / f"{fname}_surrogate.csv"
        print(f"Writing surrogate training data to {surrogate_path}")
        df_surrogate = dataset.drop(columns=["matched", "phase"])
        df_surrogate.to_csv(surrogate_path, index=False)


if __name__ == "__main__":
    import argparse

    HERE = Path(__file__).resolve().absolute().parent
    DATADIR = HERE.parent.parent.parent / "data" / "pd1-data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    datadir = args.data_dir if args.data_dir else DATADIR
    tarball = datadir / "data.tar.gz"
    process_pd1(tarball)

    # Print class names
    """
    print("Dataset names")
    for f in DATADIR.iterdir():
        if (
            f.suffix == ".csv"
            and "_matched" in str(f)
            and "phase1" in str(f)
            and not str(f).startswith("pd1")
        ):
            dataset, model, rest = str(f).split("-")
            batchsize, *_ = rest.split("_")
            dataset = dataset.replace("_", " ").title().replace(" ", "")
            model = model.replace("_", " ").title().replace(" ", "")
            print(f"PD1Tabular_{dataset}_{model}_{batchsize}")
    """
