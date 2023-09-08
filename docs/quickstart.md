# Quickstart
Make sure you first followed the [setup](setup.md) guide.

We will be using the synthetic [MFHartmann](examples/mfh.md) for this tutorial
as this requires no downloads to run.

In general, the only import you should need for generic use is just `import mfpbench`
and using [`mfpbench.get(...)`][mfpbench.get.get] to get a benchmark.

There are also some nuances when working with tabular data that should be mentioned,
see [Tabular Benchmarks](#tabular-benchmarks) for more information.


!!! note "Quick Reference"

    **Useful Properties**

    * [`.space`][mfpbench.Benchmark.space] - The space of the benchmark
    * [`.start`][mfpbench.Benchmark.start] - The starting fidelity of the benchmark
    * [`.end`][mfpbench.Benchmark.end] - The end fidelity of the benchmark
    * [`.fidelity_name`][mfpbench.Benchmark.fidelity_name] - The name of the fidelity
    * [`.table`][mfpbench.TabularBenchmark.table] - The table backing
        a [`TabularBenchmark`][mfpbench.TabularBenchmark.table].
    * [`Config`][mfpbench.Benchmark.Config] - The type of config used by the benchmark will
        be attached to the benchmark object.
    * [`Result`][mfpbench.Benchmark.Result] - The type of result used by the benchmark will
        be attached to the benchmark object.

    **Main Methods**

    * [`sample(n)`][mfpbench.Benchmark.sample] - Sample one or many configs from a benchmark
    * [`query(config, at)`][mfpbench.Benchmark.query] - Query a benchmark for a given fidelity
    * [`trajectory(config)`][mfpbench.Benchmark.trajectory] - Get the full trajectory curve of a config

    **Other**

    * [`load()`][mfpbench.Benchmark.load] - Load a benchmark into memory if not already

## Getting a benchmark
We try to make it so the normal use case of a benchmark is as simple as possible.
For this we use [`get()`][mfpbench.get.get]. Each benchmarks comes with it's own `#!python **kwargs` but
you can find them in the API documentation of `get()`.

```python exec="true" source="material-block" result="python" title="Get a benchmark" session="quickstart"
import mfpbench

benchmark = mfpbench.get("mfh3")
print(benchmark.name)
```

??? example "API"

    ::: mfpbench.get.get

??? tip "Preloading benchmarks"

    By default, benchmarks will not load in required data or surrogate models. To
    have these ready and in memory, you can pass in `#!python preload=True`.


## Properties of Benchmarks

All benchmarks inherit from [`Benchmark`][mfpbench.Benchmark] which has some useful
properties we might want to know about:

```python exec="true" source="material-block" result="python" title="Benchmark Properties" session="quickstart"
print(f"Benchmark fidelity starts at: {benchmark.start}")
print(f"Benchmark fidelity ends at: {benchmark.end}")
print(f"Benchmark fidelity is called: {benchmark.fidelity_name}")
print(f"Benchmark has conditionals: {benchmark.has_conditionals}")
print("Benchmark has the following space")
print(benchmark.space)
```

## Sampling from a benchmark
To sample from a benchmark, we use the [`sample()`][mfpbench.Benchmark.sample] method.
This method takes in a number of samples to return and returns a list of configs.

```python exec="true" source="material-block" result="python" title="Sample from a benchmark" session="quickstart"
config = benchmark.sample()
print(config)

configs = benchmark.sample(10, seed=2)
```

## Querying a benchmark
To query a benchmark, we use the [`query()`][mfpbench.Benchmark.query] method.
This method takes in a config and a fidelity to query at and returns the
[`Result`][mfpbench.Result] of the benchmark at that fidelity.
By default, this will return at the maximum fidelity but you can pass `at=`
to query at a different fidelity.

```python exec="true" source="material-block" result="python" title="Query a benchmark" session="quickstart"
value = benchmark.query(config)
print(value)

value = benchmark.query(config, at=benchmark.start)
print(value)
```

When querying a benchmark, you can get the entire trajctory curve of a config with
[`trajectory()`][mfpbench.Benchmark.trajectory]. This will be a `list[Result]`, one
for each fidelity available.

```python exec="true" source="material-block" result="python" title="Get the trajectory of a config" session="quickstart"
trajectory = benchmark.trajectory(config)
print(len(trajectory))

errors = [r.error for r in trajectory]

trajectory = benchmark.trajectory(config, frm=benchmark.start, to=benchmark.end // 2)
print(len(trajectory))
```

!!! tip

    The query and trajectory function can take in a [`Config`][mfpbench.Config] object
    or anything that looks like a mapping.


## Working with [`Config`][mfpbench.Config] objects
When interacting with a [`Benchmark`][mfpbench.Benchmark], you will always be returned
[`Config`][mfpbench.Config] objects. These contain some simple methods to make working
with them easier.

They behave like a non-mutable dictionary  so you can use them like a
non-mutable dictionary.

```python exec="true" source="material-block" result="python" session="quickstart"
config = benchmark.sample()
print("index", config["X_1"])

print("get", config.get("X_1231", 42))

for key, value in config.items():
    print(key, value)

print("contains", "X_1" in config)

print("len", len(config))

print("dict", dict(config))
```

??? tip "How is that done?"

    This is done by inheriting from python's [`Mapping`][collections.abc.Mapping]
    class and implementing it's methods, namely `__getitem__()`
    `__iter__()`, `__len__()`. You can also implement things to look like lists, containers
    and other pythonic things!


=== "`dict()`/`from_dict()`"

    [`Config.dict()`][mfpbench.Config.dict] returns a dictionary of the config. This is useful for
    working with the config in other libraries.

    ```python exec="true" source="material-block" result="python" session="quickstart"
    config = benchmark.sample()
    print(config)

    config_dict = config.dict()
    print(config_dict)

    new_config = benchmark.Config.from_dict(config_dict)
    print(new_config)
    ```

=== "`copy()`"

    [`Config.copy()`][mfpbench.Config.copy] returns a new config with the same values.

    ```python exec="true" source="material-block" result="python" session="quickstart"
    config = benchmark.sample()
    print(config)

    new_config = config.copy()
    print(new_config)
    print(new_config == config)
    ```

=== "`mutate()`"

    [`Config.mutate()`][mfpbench.Config.mutate] takes in a dictionary of keys to values
    and returns a new config with those values changed.

    ```python exec="true" source="material-block" result="python" session="quickstart"
    config = benchmark.sample()
    print(config)

    new_config = config.mutate(X_1=0.5)
    print(new_config)
    ```

=== "`perturb()`"

    [`Config.perturb()`][mfpbench.Config.perturb] takes in the space the config is from,
    a standard deviation and/or a categorical swap change and returns a new config with
    the values perturbed by a normal distribution with the given standard deviation and/or
    the categorical swap change.

    ```python exec="true" source="material-block" result="python" session="quickstart"
    config = benchmark.sample()
    print(config)

    perturbed_config = config.perturb(space=benchmark.space, std=0.2)
    print(perturbed_config)
    ```

=== "`save()`/`from_file()`"

    [`Config.save()`][mfpbench.Config.save] and [`Config.from_file()`][mfpbench.Config.from_file] are
    used to save and load configs to and from disk.

    ```python exec="true" source="material-block" result="python" session="quickstart"
    config = benchmark.sample()
    print(config)

    config.save("config.yaml")
    loaded_config = benchmark.Config.from_file("config.yaml")

    config.save("config.json")
    loaded_config = benchmark.Config.from_file("config.json")

    print(loaded_config == config)
    ```


## Working with [`Result`][mfpbench.Result] objects
When interacting with a [`Benchmark`][mfpbench.Benchmark], all results will be communicated back
with [`Result`][mfpbench.Result] objects. These contain some simple methods to make working
with them easier. Every benchmark will have a different set of results available but in general
we try to make at least an [`error`][mfpbench.Result.error] and [`score`][mfpbench.Result.score]
available. We also make a [`cost`][mfpbench.Result.cost] available for benchmarks, which
is often something like the time taken to train the specific config. These `error` and `score`
attributes are usually **validation** errors and scores. Some benchmarks also provide a `test_error`
and `test_score` which are the **test** errors and scores, but not all.

```python exec="true" source="material-block" result="python" session="quickstart"
config = benchmark.sample()
result = benchmark.query(config)

print("error", result.error)
print("cost", result.cost)

print(result)
```

These share the [`dict()`][mfpbench.Result.dict] and [`from_dict()`][mfpbench.Result.from_dict]
methods as [`Config`][mfpbench.Config] objects but do not behave like dictionaries.

The most notable property of [`Result`][mfpbench.Result] objects is that also have the
[`fidelity`][mfpbench.Result.fidelity] at which they were evaluated at and also
the [`config`][mfpbench.Config] that was evaluated to generate the results.

## Tabular Benchmarks
Some benchmarks are tabular in nature, meaning they have a table of results that
can be queried. These benchmarks inherit from [`TabularBenchmark`][mfpbench.TabularBenchmark]
and have a [`table`][mfpbench.TabularBenchmark.table] property that is the ground source of truth
for the benchmark. This table is a [`pandas.DataFrame`][pandas.DataFrame] and can be
queried as such.

In general, tabular benchmarks will have to construct themselves using the base [`TabularBenchmark`][mfpbench.TabularBenchmark]
This requires the follow arguments which can be used to normalize the table for efficient indexing and usage.
Predefined tabular benchmarks will fill these in easily for you, e.g. [`LCBenchTabularBenchmark`][mfpbench.LCBenchTabularBenchmark].

??? example "Required arguments for a `TabularBenchmark`"

    The main required arguments are [`.config_name`][mfpbench.TabularBenchmark.config_name], [`.fidelity_name`][mfpbench.TabularBenchmark.fidelity_name],
    [`.config_keys`][mfpbench.TabularBenchmark.config_keys], [`.result_keys`][mfpbench.TabularBenchmark.result_keys]

    ::: mfpbench.TabularBenchmark.__init__

### Difference for `Config`
When working with tabular benchmarks, the config type that is used is a [`TabularConfig`][mfpbench.TabularConfig].
The one difference is that it includes an [`.id`][mfpbench.TabularConfig.id] property that is used to
identify the config in the table. **This is what's used to retrieve results from the table**.
If this is missing when doing a [`query()`][mfpbench.Benchmark.query], we'll do our best to match
the config to the table and get the correct id, but this is not guaranteed.

When using [`dict()`][mfpbench.TabularConfig.dict], this `id` is **not** included in the dictionary.
In general you should either store the `config` object itself or at least `config.id`, that you can
include back in before calling [`query()`][mfpbench.Benchmark.query].

### Using your own Tabular Data
To facilitate each of use for you own usage of tabular data, we provide a
[`GenericTabularBenchmark`][mfpbench.GenericTabularBenchmark] that can be used
to load in and use your own tabular data.

```python exec="true" source="material-block" result="python" session="generic tabular"
import pandas as pd
from mfpbench import GenericTabularBenchmark

# Create some fake data
df = pd.DataFrame(
    {
        "config": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        "fidelity": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "balanced_accuracy": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "color": ["red", "red", "red", "blue", "blue", "blue", "green", "green", "green"],
        "shape": ["circle", "circle", "circle", "square", "square", "square", "triangle", "triangle", "triangle"],
        "kind": ["mlp", "mlp", "mlp", "mlp", "mlp", "mlp", "mlp", "mlp", "mlp"],
    }
)
print(df)
print()
print()

benchmark = GenericTabularBenchmark(
    df,
    name="mydata",
    config_name="config",
    fidelity_name="fidelity",
    config_keys=["color", "shape"],
    result_keys=["balanced_accuracy"],
    result_mapping={
        "error": lambda df: 1 - df["balanced_accuracy"],
        "score": lambda df: df["balanced_accuracy"],
    },
    remove_constants=True,
)

print(benchmark.table)
```

You can then operate on this benchmark as expected.

```python exec="true" source="material-block" result="python" session="generic tabular"
config = benchmark.sample()
print(config)

result = benchmark.query(config, at=2)

print(result)
```

??? example "API for `GenericTabularBenchmark`"

    ::: mfpbench.GenericTabularBenchmark.__init__
