# mf-prior-bench

## Installation
```bash
git clone git@github.com:automl/mf-prior-bench.git
cd mf-prior-bench

# Create your env however
...

# I use pip and virtualenv
pip install -e  ".[dev]"
# pipe install -e ".[dev, jahs-bench]  # If requiring jahs bench

# Pre-commit
pre-commit install

# Optionally there's a makefile command for that
make install-dev  # use `make help` if you like

# You're going to want to download things needed jahs-bench, YAHPO, and pd1
# Just leave --data-dir at default to be safest  "./data/{jahs-bench-data, yahpo-gym-data, pb1-data}"
python -m mfpbench download [--force] [--data-dir PATH]
```

### Usage
For a more indepth usage and runnalbe example, use `example1.py`.


#### Benchmark
For the most simplistic usage:
```python
import mfpbench

bench = mfpbench.get("lcbench", task_id="3945", seed=1)

config = benchmark.sample()
result = benchmark.query(config, at=200)

print(result.error)

trajectory = benchmark.trajectory(config)
print(trajectory)
```

Some more general properties available on every benchmark
```python
# The name of the fidelity used by the benchmark, i.e. "epoch", "trainsize"
bench.fidelity_name: str

# The start, stop, step of the fidelities
bench.fidelity_range: tuple[int, int, int] | tuple[float, float, float]

# The configuration space of the benchmark
bench.space: ConfigurationSpace

# Whether the benchmark has conditional hyperparameters
bench.has_conditionals: bool

# Preload the benchmark if needed, otherwise it loads on demand
bench.load()

# If you need to iterthe fidelities of the benchmark
for f in bench.iter_fidelities():
    ...
```

#### Config
Configurations for the benchmark typically come from a `ConfigurationSpace` but we pass around
type safe `Config` objects for every benchmark.
```python
# Sampling
config: Config = bench.sample()
configs: list[Config] = bench.sample(10)

# Immutable but provide convenience methods
config_other = config.copy()
assert config is not config_other  # Different objects
assert config == config_other      # But they are equal

# They are hashable as dict or set entries
samples = bench.sample(50)
unique_samples = set(samples)

# You can mutate them if you need
config_other = config.mutate(kwarg=value)

# You can always validate the config to make sure it's valid
config.validate()

# You can create a config from a dictionary or a ConfigSpace Configuration with:
d = { ... }
config.from_dict(d)

configspace_configuration = bench.space.sample_configuration(1)
config.from_dict(configspace_configuration)
````

#### Result
When doing `bench.query(config)` you will get back a specific `Result` for that benchmark.
This will also include the `config` that generated this result.

Each benchmark will have return different kinds of results, like `validation_acc`, `traintime`, `mem_size` etc...
For this reason, we provide some general properties available for all benchmarks as seen in the snippet below.

Importantly though, the fidelity is always just `result.fidelity` and we will always have an associated `result.cost` which is akin to training time for an algorithm or some monotonically increasing value with respect to the fidelity. _(Note: surrogate benchmarks don't gaurantee this in general but is approximately true)_

Some will also return a metric that is to be maximised while other provide a minimize target.
For this reason we provide a `score` and `error` which are inverses of each other. For example, if the
`score` is accuracy, `error = 1 - score`.
For unbounded metrics, we use a sign-flip to have both a `score` and `error`.

Where `test` and `val` scores are available, these are available through `test_score`, `test_error`, `val_score`
and `val_error`. When these are not available, we still keep these populated but they will both be the same
as just `score` and `error`.

Not all benchmarks return the same kind of information and they are under differnt names so we provide
some gauranteed properties:
```python
result = bench.query(config)

# These are inverse of each other, see above
result.score
result.error

# The fidelity and config that generated the result
result.fidelity
result.config

# test and val score/errors, see above
result.val_score
result.val_error
result.val_score
result.val_error

# Just get it as a raw dict
d = result.dict()

# They are hashable and keep equality
result1 = bench.query(config)
result2 = bench.query(config)

assert result1 == result2

# If you ever need to manually create one
result = bench.Result.from_dict(
    config=config,
    fidelity=fidelitiy,
    result={...}
)
```

### NOTE
Test command to train
```bash
python -m mfpbench.pd1.surrogate.training --dataset="lm1b-transformer-2048" --seed 1 --cv 2 --time 30 --y "valid_error_rate"
python -m mfpbench.pd1.surrogate.training --dataset="lm1b-transformer-2048" --seed 1 --cv 2 --time 30 --y "train_cost"
```

### Contributing
TODO
