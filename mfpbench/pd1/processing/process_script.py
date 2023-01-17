from __future__ import annotations

import gzip
import json
import logging
import shutil
from dataclasses import dataclass
from itertools import accumulate, product
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from mfpbench.pd1.processing.columns import COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_accumulate(
    x: Iterator[float | None] | float, fill: float = np.nan
) -> Iterator[float]:
    """Accumulate, but fill in missing values with a default value."""
    if isinstance(x, float):
        return iter([fill if np.isnan(x) else x])

    itr = iter(f if f is not None else fill for f in x)
    return accumulate(itr)


def uniref50_epoch_convert(x: float | list[float]) -> float | list[float]:
    """Converts the epochs of uniref50 to some usable form.

    Converts:
        0             NaN
        1    [0, 0, 0, 1]
        2           [nan]
        3     [0, 0, nan].

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

        if not self.unpacked_path.exists():
            logger.info(f"Unpacking from {frm}")
            with gzip.open(frm, mode="rt") as f:
                data = [json.loads(line) for line in f]
            unpacked = pd.DataFrame(data)

            logger.info(f"Saving to {self.unpacked_path}")
            unpacked.to_csv(
                self.unpacked_path,
                index=False,
            )
        else:
            logger.info(f"Unpacking from {frm}")
            unpacked = pd.read_csv(self.unpacked_path)
            assert isinstance(unpacked, pd.DataFrame)
            columns = {col.name: col for col in COLUMNS}

            # Convert a string representing a list to a
            # real list of the contained objects using json.loads
            list_columns = [
                col
                for col in unpacked.columns
                if col in columns and columns[col].type is list
            ]
            for c in list_columns:
                unpacked[c] = [
                    json.loads(val.replace("None", "null"))
                    if isinstance(val, str)
                    else val
                    for _, val in unpacked[c].items()  # type: ignore
                ]

            assert isinstance(unpacked, pd.DataFrame)

        return unpacked

    @property
    def unpacked_path(self) -> Path:
        return self.dir / f"{self.rawname}_unpacked.csv"


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
        readme_path = rawdir / "README.txt"
        if not readme_path.exists():
            shutil.unpack_archive(tarball, rawdir)

            unpacked_folder_name = "pd1"  # This is what the tarball will unpack into
            unpacked_folder = rawdir / unpacked_folder_name

            # Move everything from the uncpack folder to the "raw" folder
            for filepath in unpacked_folder.iterdir():
                to = rawdir / filepath.name
                shutil.move(str(filepath), str(to))

            # Remove the archive folder, its all been moved to "raw"
            shutil.rmtree(str(unpacked_folder))

    # For processing the df
    drop_columns = [c.name for c in COLUMNS if not c.keep]
    renames = {c.name: c.rename for c in COLUMNS if c.rename is not None}
    hps = [c.rename for c in COLUMNS if c.hp]
    # metrics = [c.rename if c.rename else c.name for c in COLUMNS if c.metric]

    dfs: list[pd.DataFrame] = []
    for matched, phase in product([True, False], [0, 1]):
        # Unpack the jsonl.gz archive if needed
        datapack = Datapack(matched=matched, phase=phase, dir=rawdir)
        df = datapack.unpack()

        # Tag them from the dataset they came from
        df["matched"] = matched
        df["phase"] = phase

        df.drop(columns=drop_columns, inplace=True)
        df.rename(columns=renames, inplace=True)

        dfs.append(df)

    # We now merge them all into one super table for convenience
    full_df = pd.concat(dfs, ignore_index=True)

    # Since some columns values are essentially lists, we need to explode them out
    # However, we need to make sure to distuinguish between transformer and not as
    # transformers do not have test error available
    # We've already renamed the columns them at this point
    list_columns = [c.rename if c.rename else c.name for c in COLUMNS if c.type == list]
    transformer_datasets = ["uniref50", "translate_wmt", "imagenet", "lm1b"]
    dataset_columns = ["dataset", "model", "batch_size"]

    groups = full_df.groupby(dataset_columns)
    for (name, model, batchsize), dataset in groups:  # type: ignore
        fname = f"{name}-{model}-{batchsize}"
        logger.info(fname)

        if name in transformer_datasets:
            explode_columns = [c for c in list_columns if c != "test_error_rate"]
            dataset.drop(columns=["test_error_rate"], inplace=True)
        else:
            explode_columns = list_columns

        if name == "uniref50":
            # For some reason the epochs of this datasets are basically [0, 0, 0, 1]
            # We just turn this into an incremental thing
            epochs = dataset["epoch"]
            assert epochs is not None
            dataset["epoch"] = dataset["epoch"].apply(  # type: ignore
                uniref50_epoch_convert
            )

        # Make sure train_cost rows are all of equal length
        dataset["train_cost"] = [
            np.nan if r in (None, np.nan) else list(safe_accumulate(r, fill=np.nan))
            for r in dataset["train_cost"]  # type: ignore
        ]

        # Explode out the lists in the entires of the datamframe to be a single long
        # dataframe with each element of that list on its own row
        dataset = dataset.explode(explode_columns, ignore_index=True)
        logger.info(f"{len(dataset)} rows")
        assert isinstance(dataset, pd.DataFrame)

        # Remove any rows that have a nan in the exploded columns
        nan_rows = dataset["train_cost"].isna()
        logger.info(f" - len(nan_rows) {sum(nan_rows)}")

        logger.debug(f"Removing rows with nan in {explode_columns}")
        dataset = dataset[~nan_rows]  # type: ignore
        assert isinstance(dataset, pd.DataFrame)

        logger.info(f"{len(dataset)} rows (after nan removal)")

        if fname == "lm1b-transformer-2048":
            # Some train costs go obsenly high for no reason, we drop these rows
            dataset = dataset[dataset["train_cost"] < 10_000]  # type: ignore

        elif fname == "uniref50-transformer-128":
            # Some train costs go obsenly high for no reason, we drop these rows
            # Almost all are below 400 but we add a buffer
            dataset = dataset[dataset["train_cost"] < 4_000]  # type: ignore

        elif fname == "imagenet-resnet-512":
            # We drop all configs that exceed the 0.95 quantile in their max train_cost
            # as we consider this to be a diverging config. The surrogate will smooth
            # out these configs as it is not aware of divergence
            # NOTE: q95 was experimentally determined so as to not remove too many
            # configs but remove configs which would create massive gaps in "train_cost"
            # which would cause optimization of the surrogate to focus too much on
            # minimizing it's loss for outliers
            hp_names = ["lr_decay_factor", "lr_initial", "lr_power", "opt_momentum"]
            maxes = [
                v["train_cost"].max()  # type: ignore
                for _, v in dataset.groupby(hp_names)
            ]
            q95 = np.quantile(maxes, 0.95)
            configs_who_dont_exceed_q95 = (
                v
                for _, v in dataset.groupby(hp_names)
                if v["train_cost"].max() < q95  # type: ignore
            )
            dataset = pd.concat(configs_who_dont_exceed_q95, axis=0)

        elif fname == "cifar100-wide_resnet-2048":
            # We drop all configs that exceed the 0.95 quantile in their max train_cost
            # as we consider this to be a diverging config. The surrogate will smooth
            # out these configs as it is not aware of divergence
            # NOTE: q93 was experimentally determined so as to not remove too many
            # configs but remove configs which would create massive gaps in
            # "train_cost" which would cause optimization of the surrogate to
            # focus too much on minimizing it's loss for outliers
            hp_names = ["lr_decay_factor", "lr_initial", "lr_power", "opt_momentum"]
            maxes = [
                v["train_cost"].max()  # type: ignore
                for _, v in dataset.groupby(hp_names)
            ]
            q93 = np.quantile(maxes, 0.93)
            configs_who_dont_exceed_q93 = (
                v
                for _, v in dataset.groupby(hp_names)
                if v["train_cost"].max() < q93  # type: ignore
            )
            dataset = pd.concat(configs_who_dont_exceed_q93, axis=0)

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

        dataset = dataset.drop(columns=drop_columns)  # type: ignore

        if handle_nans == "fill":
            raise NotImplementedError("TODO")

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
            hps = list(hps + ["activation_fn"])
        else:
            hps = list(hps)

        dataset = dataset.drop_duplicates(hps + ["epoch"], keep="last")  # type: ignore

        # The rest can be used for surrogate training
        surrogate_path = datadir / f"{fname}_surrogate.csv"
        df_surrogate = dataset.drop(columns=["matched", "phase"])
        df_surrogate.to_csv(surrogate_path, index=False)


if __name__ == "__main__":
    import argparse

    HERE = Path(__file__).resolve().absolute().parent
    DATADIR = HERE.parent.parent.parent / "data" / "pd1-data"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=DATADIR, help="Where the data directory is"
    )
    args = parser.parse_args()

    # Print class names
    for f in args.datadir.iterdir():
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

    tarball = args.data_dir / "data.tar.gz"
    process_pd1(tarball)
