from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar, overload
from typing_extensions import override

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from more_itertools import first_true

from mfpbench.benchmark import Benchmark
from mfpbench.config import GenericTabularConfig, TabularConfig
from mfpbench.result import GenericTabularResult, Result

# The kind of Config to the **tabular** benchmark
CTabular = TypeVar("CTabular", bound=TabularConfig)

# The return value from a config query
R = TypeVar("R", bound=Result)

# The kind of fidelity used in the benchmark
F = TypeVar("F", int, float)


class TabularBenchmark(Benchmark[CTabular, R, F]):
    id_key: str
    """The column in the table that contains the config id. Will be set to the index"""

    fidelity_key: str
    """The name of the fidelity used in this benchmark"""

    config_keys: Sequence[str]
    """The keys in the table that contain the config"""

    result_keys: Sequence[str]
    """The keys in the table that contain the results"""

    table: pd.DataFrame
    """The table of results used for this benchmark"""

    configs: Mapping[str, CTabular]
    """The configs used in this benchmark"""

    # The config and result type of this benchmark
    Config: type[CTabular]
    Result: type[R]

    # Whether this benchmark has conditonals in it or not
    has_conditionals: bool = False

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        table: pd.DataFrame,
        *,
        id_key: str,
        fidelity_key: str,
        result_keys: Sequence[str],
        config_keys: Sequence[str],
        remove_constants: bool = False,
        space: ConfigurationSpace | None = None,
        seed: int | None = None,
        prior: str | Path | CTabular | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
    ):
        """Initialize the benchmark.

        Args:
            name: The name of this benchmark.
            table: The table to use for the benchmark.
            id_key: The column in the table that contains the config id
            fidelity_key: The column in the table that contains the fidelity
            result_keys: The columns in the table that contain the results
            config_keys: The columns in the table that contain the config values
            remove_constants: Remove constant config columns from the data or not.
            space: The configuration space to use for the benchmark. If None, will
                just be an empty space.
            prior: The prior to use for the benchmark. If None, no prior is used.
                If a string, will be treated as a prior specific for this benchmark
                if it can be found, otherwise assumes it to be a Path.
                If a Path, will load the prior from the path.
                If a dict or Configuration, will be used directly.
            perturb_prior: If not None, will perturb the prior by this amount.
                For numericals, while for categoricals, this is interpreted as the
                probability of swapping the value for a random one.
            seed: The seed to use for the benchmark.
        """
        cls = self.__class__

        # Make sure we work with a clean slate, no issue with index.
        table = table.reset_index()

        # Make sure all the keys they specified exist
        if id_key not in table.columns:
            raise ValueError(f"'{id_key=}' not in columns {table.columns}")

        if fidelity_key not in table.columns:
            raise ValueError(f"'{fidelity_key=}' not in columns {table.columns}")

        if not all(key in table.columns for key in result_keys):
            raise ValueError(f"{result_keys=} not in columns {table.columns}")

        if not all(key in table.columns for key in config_keys):
            raise ValueError(f"{config_keys=} not in columns {table.columns}")

        # Make sure that the column `id` only exist if it's the `id_key`
        if "id" in table.columns and id_key != "id":
            raise ValueError(
                f"Can't have `id` in the columns if it's not the {id_key=}."
                " Please drop it or rename it.",
            )

        # Remove constants from the table
        if remove_constants:

            def is_constant(_s: pd.Series) -> bool:
                _arr = _s.to_numpy()
                return bool((_arr == _arr[0]).all())

            constant_cols = [
                col for col in table.columns if is_constant(table[col])  # type: ignore
            ]
            table = table.drop(columns=constant_cols)  # type: ignore
            config_keys = [k for k in config_keys if k not in constant_cols]

        # Remap their id column to `id`
        table = table.rename(columns={id_key: "id"})

        # Index the table
        index_cols: list[str] = ["id", fidelity_key]

        # Drop all the columns that are not relevant
        relevant_cols: list[str] = [
            *index_cols,
            *result_keys,
            *config_keys,
        ]
        table = table[relevant_cols]  # type: ignore
        table = table.set_index(index_cols).sort_index()

        # We now have the following table
        #
        #     id    fidelity | **metric, **config_values
        #     0         0    |
        #               1    |
        #               2    |
        #     1         0    |
        #               1    |
        #               2    |
        #   ...

        # Make sure we have equidistance fidelities for all configs
        fidelity_values = table.index.get_level_values(fidelity_key)
        fidelity_counts = fidelity_values.value_counts()
        if not (fidelity_counts == fidelity_counts.iloc[0]).all():
            raise ValueError(f"{fidelity_key=} not uniform. \n{fidelity_counts}")

        sorted_fids = sorted(fidelity_values.unique())
        start = sorted_fids[0]
        end = sorted_fids[-1]
        step = sorted_fids[1] - sorted_fids[0]

        # Here we get all the unique configs
        #     id    fidelity | **metric, **config_values
        #     0         0    |
        #     1         0    |
        #   ...
        id_table = table.groupby(level="id").agg("first")
        configs = {
            str(config_id): cls.Config.from_dict(
                {
                    **row[config_keys].to_dict(),  # type: ignore
                    "id": str(config_id),
                },
            )
            for config_id, row in id_table.iterrows()
        }

        # Create the configuration space
        if space is None:
            space = ConfigurationSpace(name, seed=seed)

        self.table = table
        self.configs = configs
        self.fidelity_key = fidelity_key
        self.id_key = id_key
        self.config_keys = sorted(config_keys)
        self.result_keys = sorted(result_keys)
        self.fidelity_range = (start, end, step)  # type: ignore

        super().__init__(
            name=name,
            seed=seed,
            space=space,
            prior=prior,
            perturb_prior=perturb_prior,
        )

    def query(
        self,
        config: CTabular | Mapping[str, Any] | str,
        at: F | None = None,
        *,
        argmax: str | None = None,
        argmin: str | None = None,
    ) -> R:
        """Submit a query and get a result.

        !!! warning "Passing a raw config"

            If a mapping is passed (and **not** a [`Config`][mfpbench.Config] object),
            we will attempt to look for `id` in the mapping, to know which config to
            lookup.

            If this fails, we will try match the config to one of the configs in
            the benchmark.

            Prefer to pass the [`Config`][mfpbench.Config] object directly if possible.

        ??? note "Override"

            This function overrides the default
            [`query()`][mfpbench.Benchmark.query] to allow for this
            config matching

        Args:
            config: The query to use
            at: The fidelity at which to query, defaults to None which means *maximum*
            argmax: Whether to return the argmax up to the point `at`. Will be slower as
                it has to get the entire trajectory. Uses the key from the Results.
            argmin: Whether to return the argmin up to the point `at`. Will be slower as
                it has to get the entire trajectory. Uses the key from the Results.

        Returns:
            The result of the query
        """
        _config = self._find_config(config)
        return super().query(
            _config,
            at=at,  # type: ignore
            argmax=argmax,
            argmin=argmin,
        )

    @override
    def trajectory(
        self,
        config: CTabular | Mapping[str, Any] | str,
        *,
        frm: F | None = None,
        to: F | None = None,
        step: F | None = None,
    ) -> list[R]:
        """Submit a query and get a result.

        !!! warning "Passing a raw config"

            If a mapping is passed (and **not** a [`Config`][mfpbench.Config] object),
            we will attempt to look for `id` in the mapping, to know which config to
            lookup.

            If this fails, we will try match the config to one of the configs in
            the benchmark.

            Prefer to pass the [`Config`][mfpbench.Config] object directly if possible.

        ??? note "Override"

            This function overrides the default
            [`trajectory()`][mfpbench.Benchmark.trajectory] to allow for this
            config matching

        Args:
            config: The query to use
            frm: Start of the curve, should default to the start
            to: End of the curve, should default to the total
            step: Step size, defaults to ``cls.default_step``

        Returns:
            The result of the query
        """
        _config = self._find_config(config)
        return super().trajectory(_config, frm=frm, to=to, step=step)  # type: ignore

    def _find_config(
        self,
        config: CTabular | Mapping[str, Any] | str | int,
    ) -> CTabular:
        # It's an interger but likely meant to be string
        # We don't do any numeric based lookups
        if isinstance(config, int):
            config = str(config)

        # It's a key into the self.configs dict
        if isinstance(config, str):
            return self.configs[config]

        # If's a Config, that's fine
        if isinstance(config, self.Config):
            return config

        # At this point, we assume we're basically dealing with a dictionary
        assert isinstance(config, Mapping)

        # Not sure how that ended up there, but we can at least handle that
        if self.id_key in config:
            _real_config_id = str(config[self.id_key])
            return self.configs[_real_config_id]

        # Also ... not sure but anywho
        if "id" in config:
            _id = config["id"]
            return self.configs[_id]

        # Alright, nothing worked, here we try to match the actual hyperparameter
        # values to what we have in our known configs and attempt to get the
        # id that way
        match = first_true(
            self.configs.values(),
            pred=lambda c: c == config,  # type: ignore
            default=None,
        )
        if match is None:
            raise ValueError(
                f"Could not find config matching {config}. Please pass the"
                f" `Config` object or specify the `id` in the {type(config)}",
            )
        return match

    @override
    def _objective_function(self, config: CTabular, at: F) -> R:
        """Submit a query and get a result.

        Args:
            config: The query to use
            at: The fidelity at which to query

        Returns:
            The result of the query
        """
        row = self.table.loc[(config.id, at)]

        row.name = config.id
        config = self.Config.from_row(row[self.config_keys])
        results = row[self.result_keys]
        return self.Result.from_row(config=config, row=results, fidelity=at)

    # No number specified, just return one config
    @overload
    def sample(
        self,
        n: None = None,
        *,
        seed: int | np.random.RandomState | None = ...,
    ) -> CTabular:
        ...

    # With a number, return many in a list
    @overload
    def sample(
        self,
        n: int,
        *,
        seed: int | np.random.RandomState | None = ...,
    ) -> list[CTabular]:
        ...

    @override
    def sample(
        self,
        n: int | None = None,
        *,
        seed: int | np.random.RandomState | None = None,
    ) -> CTabular | list[CTabular]:
        """Sample a random possible config.

        Args:
            n: How many samples to take, None means jsut a single one, not in a list
            seed: The seed to use for the sampling.

                !!! note "Seeding"

                    This is different than any seed passed to the construction
                    of the benchmark.

        Returns:
            Get back a possible Config to use
        """
        _seed: int | None
        if isinstance(seed, np.random.RandomState):
            _seed = seed.random_integers(0, 2**31 - 1)
        else:
            _seed = seed

        rng = np.random.default_rng(seed=_seed)

        config_items: list[CTabular] = list(self.configs.values())
        n_configs = len(config_items)
        sample_amount = n if n is not None else 1

        if sample_amount > n_configs:
            raise ValueError(
                f"Can't sample {sample_amount} configs from {n_configs} configs",
            )

        indices = rng.choice(n_configs, size=sample_amount, replace=False)
        if n is None:
            first_index: int = indices[0]
            return config_items[first_index]

        return [config_items[i] for i in indices]


class GenericTabularBenchmark(
    TabularBenchmark[
        GenericTabularConfig,
        GenericTabularResult[GenericTabularConfig, F],
        F,
    ],
):
    Result = GenericTabularResult
    Config = GenericTabularConfig

    def __init__(  # noqa: PLR0913
        self,
        table: pd.DataFrame,
        *,
        name: str | None = None,
        id_key: str,
        fidelity_key: str,
        result_keys: Sequence[str],
        config_keys: Sequence[str],
        result_mapping: (dict[str, str | Callable[[pd.DataFrame], Any]] | None) = None,
        remove_constants: bool = False,
        space: ConfigurationSpace | None = None,
        seed: int | None = None,
        prior: str | Path | GenericTabularConfig | Mapping[str, Any] | None = None,
        perturb_prior: float | None = None,
    ):
        """Initialize the benchmark.

        Args:
            table: The table to use for the benchmark
            name: The name of the benchmark. If None, will be set to
                `unknown-{datetime.now().isoformat()}`
            id_key: The column in the table that contains the config id
            fidelity_key: The column in the table that contains the fidelity
            result_keys: The columns in the table that contain the results
            config_keys: The columns in the table that contain the config values
            result_mapping: A mapping from the result keys to the table keys.
                If a string, will be used as the key in the table. If a callable,
                will be called with the table and the result will be used as the value.
            remove_constants: Remove constant config columns from the data or not.
            space: The configuration space to use for the benchmark. If None, will
                just be an empty space.
            seed: The seed to use.
            prior: The prior to use for the benchmark. If None, no prior is used.
                If a str, will check the local location first for a prior
                specific for this benchmark, otherwise assumes it to be a Path.
                If a Path, will load the prior from the path.
                If a Mapping, will be used directly.
            perturb_prior: If not None, will perturb the prior by this amount.
                For numericals, this is interpreted as the standard deviation of a
                normal distribution while for categoricals, this is interpreted
                as the probability of swapping the value for a random one.
        """
        if name is None:
            name = f"unknown-{datetime.now().isoformat()}"

        _result_mapping: dict = result_mapping if result_mapping is not None else {}

        # Remap the result keys so it works with the generic result types
        if _result_mapping is not None:
            for k, v in _result_mapping.items():
                if isinstance(v, str):
                    if v not in table.columns:
                        raise ValueError(f"{v} not in columns\n{table.columns}")

                    table[k] = table[v]
                elif callable(v):
                    table[k] = v(table)
                else:
                    raise ValueError(f"Unknown result mapping {v} for {k}")

        super().__init__(
            name=name,
            table=table,
            id_key=id_key,
            fidelity_key=fidelity_key,
            result_keys=[*result_keys, *_result_mapping.keys()],
            config_keys=config_keys,
            remove_constants=remove_constants,
            space=space,
            seed=seed,
            prior=prior,
            perturb_prior=perturb_prior,
        )


if __name__ == "__main__":
    HERE = Path(__file__).parent
    path = HERE.parent.parent / "data" / "lcbench-tabular" / "adult.parquet"
    table = pd.read_parquet(path)
    benchmark = GenericTabularBenchmark(
        table=table,
        id_key="id",
        fidelity_key="epoch",
        result_keys=[
            "time",
            "val_accuracy",
            "val_cross_entropy",
            "val_balanced_accuracy",
            "test_accuracy",
            "test_cross_entropy",
            "test_balanced_accuracy",
        ],
        result_mapping={
            "error": lambda df: 1 - df["val_accuracy"],
            "score": lambda df: df["val_accuracy"],
        },
        config_keys=[
            "batch_size",
            "loss",
            "imputation_strategy",
            "learning_rate_scheduler",
            "network",
            "max_dropout",
            "normalization_strategy",
            "optimizer",
            "cosine_annealing_T_max",
            "cosine_annealing_eta_min",
            "activation",
            "max_units",
            "mlp_shape",
            "num_layers",
            "learning_rate",
            "momentum",
            "weight_decay",
        ],
        remove_constants=True,
    )
    # benchmark = LCBenchTabular(task="adult")
    all_configs = benchmark.configs  # type: ignore
    config_ids = list(all_configs.keys())
    configs = list(all_configs.values())

    config = benchmark.sample(seed=1)
    config_id = config.id

    result = benchmark.query(config, at=1)
    argmin_score = benchmark.query(config, at=42, argmin="error")

    trajectory = benchmark.trajectory(config, frm=1, to=10)

    # lcbench = LCBenchTabular(task="adult")
    # All the same stuff as above
