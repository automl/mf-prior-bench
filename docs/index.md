# `mfpbench`

Welcome to the documentation of the Multi-Fidelity Priors Benchmark library.
Check out the [quickstart](quickstart.md) to get started or check out our examples.

At a glance, this library wraps several benchmarks that give access to the **multi-fidelity**,
**learning curve** information of **configurations** from a static **configuration space**,
as well as allow you to set the default configuration of some search space as the **prior**.
This mainly includes **validation score/error** as well as **cost**.

This includes any necessary setup commands which you can find in the [setup](setup.md).

??? example "Terminology? Deep Learning Example"

    > In the deep learning setting, we often train configurations of a Neural Network
    for several epochs, recording the performance as it goes.
    However, finding the _best_ configurations requires searching over some possible set
    Every configuration is going to take a different amount of time to train but often
    you'd like to find the best configuration as fast as possible.

    > You start with your intutition and choose `#!python learning_rate=0.001` and
    apply some `#!python augmentation=True`, however when searching you also allow
    different values. You split your data into `#!python "train", "val", "test"` sets,
    and set up your pipeline so your network trains on `#!python "train"`,
    you score how good the hyperparameters are on `#!python "val"` to choose the best
    but you also evaluate these configurations on `#!python "test"`. You do this last part
    to get a sense of how much you are overfitting.

    * **fidelity** - Here this would be the _epochs_
    * **configuration** - One particular Neural Network architecture and it's hyperparameters.
    * **configuration space** - The space of possible values for all the hyperparameters, or
        in some cases, a finit set of distinct configurations.
    * **prior** - The default configuration of `#!python learning_rate=0.001` and
        `#!python augmentation=True`.
    * **validation score** - How well your model does on the `#!python "val"` set.
    * **cost** - The amount of time it took to train a model
    * **learning curve** - The ordered sequence of results for one configuration for
        each fidelity value.

```python exec="true" source="material-block" result="python" title="Example"
import mfpbench

benchmark = mfpbench.get("mfh3_good", bias=2.5, noise=0, prior="good")

config = benchmark.sample()
result = benchmark.query(config, at=34)

trajectory = benchmark.trajectory(config, frm=5, to=benchmark.end)

print(config)
print(result)
print(len(trajectory))
```

#### Benchmarks

-   [**pd1**](examples/pd1.md): A set of optimization results for larger deep-learning models from the
    [HyperBO](https://arxiv.org/abs/2109.08215) paper. We use this raw tabular data and
    build surrogate xgboost models to provide a smooth continuious space.

-   [**yahpo-gym**](examples/yahpo-gym.md) A collection of surrogate models trained from evaluations
    of various models and search spaces over a variety of models and tasks.

-   [**MFHartmann**](examples/mfh.md): A synthetic benchmark for multifidelity optimization
    that allows you to control the noise and the crossing of learning curves. Useful for
    testing the capabilities of optimizers.

-   [**LCBenchTabular**](examples/lcbench-tabular.md): A set of tabular benchmarks from
    [LCBench](https://github.com/automl/LCBench), which train various MLP's with
    [AutoPyTorch](https://github.com/automl/Auto-PyTorch) on different
    [OpenML](https://www.openml.org/) tasks.

-   **Your own**: There are also options to easily integrate your own benchmarks,
    whether [from raw tables](examples/integrating_tabular_benchmark.md)
    or with a [more sophisticated objective function](examples/integrating_objective_function_benchmark.md).
