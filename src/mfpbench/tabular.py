from __future__ import annotations

from datetime import datetime
from itertools import chain
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
    config_name: str
    """The column in the table that contains the config id. Will be set to the index"""

    fidelity_name: str
    """The name of the fidelity used in this benchmark"""

    config_keys: Sequence[str]
    """The keys in the table that contain the config"""

    result_keys: Sequence[str]
    """The keys in the table that contain the results"""

    table: pd.DataFrame
    """The table of results used for this benchmark"""

    # The config and result type of this benchmark
    Config: type[CTabular]
    Result: type[R]

    # Whether this benchmark has conditonals in it or not
    has_conditionals: bool = False

    def __init__(  # noqa: PLR0913, C901
        self,
        name: str,
        table: pd.DataFrame,
        *,
        config_name: str,
        fidelity_name: str,
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
            config_name: The column in the table that contains the config id
            fidelity_name: The column in the table that contains the fidelity
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
        if remove_constants:

            def is_constant(_s: pd.Series) -> bool:
                _arr = _s.to_numpy()
                return bool((_arr == _arr[0]).all())

            constant_cols = [
                col for col in table.columns if is_constant(table[col])  # type: ignore
            ]
            table = table.drop(columns=constant_cols)  # type: ignore
            config_keys = [k for k in config_keys if k not in constant_cols]

        # If the table isn't indexed, index it
        index_cols = [config_name, fidelity_name]
        if table.index.names != index_cols:
            # Only drop the index if it's not relevant.
            relevant_cols: list[str] = [  # type: ignore
                *list(index_cols),  # type: ignore
                *list(result_keys),
                *list(config_keys),
            ]
            relevant = any(name in relevant_cols for name in table.index.names)
            table = table.reset_index(drop=not relevant)

            if config_name not in table.columns:
                raise ValueError(f"{config_name=} not in columns {table.columns}")
            if fidelity_name not in table.columns:
                raise ValueError(f"{fidelity_name=} not in columns {table.columns}")

            table = table.set_index(index_cols)
            table = table.sort_index()

        # Make sure all keys are in the table
        for key in chain(result_keys, config_keys):
            if key not in table.columns:
                raise ValueError(f"{key=} not in columns {table.columns}")

        # Make sure the keyword "id" is not in the columns as we use it to
        # identify configs
        if "id" in table.columns:
            raise ValueError(f"{table.columns=} contains 'id'. Please rename it")

        # Make sure we have equidistance fidelities for all configs
        fidelity_values = table.index.get_level_values(fidelity_name)
        fidelity_counts = fidelity_values.value_counts()
        if not (fidelity_counts == fidelity_counts.iloc[0]).all():
            raise ValueError(f"{fidelity_name=} not  uniform. \n{fidelity_counts}")

        # We now have the following table
        #
        # config_id fidelity | **metric, **config_values
        #     0         0    |
        #               1    |
        #               2    |
        #     1         0    |
        #               1    |
        #               2    |
        #   ...

        # Here we get all the unique configs
        # config_id fidelity | **metric, **config_values
        #     0         0    |
        #     1         0    |
        #   ...
        config_id_table = table.groupby(level=config_name).agg("first")
        configs = {
            str(config_id): cls.Config.from_dict(
                {
                    **row[config_keys].to_dict(),  # type: ignore
                    "id": str(config_id),
                },
            )
            for config_id, row in config_id_table.iterrows()
        }

        fidelity_values = table.index.get_level_values(fidelity_name).unique()

        # We just assume equidistant fidelities
        sorted_fids = sorted(fidelity_values)
        start = sorted_fids[0]
        end = sorted_fids[-1]
        step = sorted_fids[1] - sorted_fids[0]

        # Create the configuration space
        if space is None:
            space = ConfigurationSpace(name, seed=seed)

        self.table = table
        self.configs = configs
        self.fidelity_name = fidelity_name
        self.config_name = config_name
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
        if isinstance(config, int):
            config = str(config)

        if isinstance(config, str):
            return self.configs[config]

        if isinstance(config, self.Config):
            return config

        if self.config_name in config:
            _id = config[self.config_name]
            return self.configs[_id]

        if "id" in config:
            _id = config["id"]
            return self.configs[_id]

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
            _seed = seed.random_integers(0, 2**32 - 1)
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
        fidelity_name: str,
        config_name: str,
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

            fidelity_name: The column in the table that contains the fidelity
            config_name: The column in the table that contains the config id
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
            config_name=config_name,
            fidelity_name=fidelity_name,
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
        fidelity_name="epoch",
        config_name="config_id",
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
