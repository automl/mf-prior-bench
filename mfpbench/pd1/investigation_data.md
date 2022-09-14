### From reading README
Below we have extracted this info from the README which informs how we process the data

> Matched data used the same set of uniformly-sampled (Halton sequence) hyperparameter
points across all tasks and unmatched data sampled new points for each task.
All other training pipeline hyperparameters were fixed to hand-selected, task-specific
default values that are described in the various task-specific .json config files that
are also included.
Since all the neural networks were trained using the code at
https://github.com/google/init2winit,
that repository is the best source for understanding the precise semantics of any
hyperparameter or task-specific configuration parameter.
The JSON config files should be human readable, minimally-redacted examples of the
actual configuration files used to create the data.
Every pair of dataset and model used in the paper has a corresponding JSON config example,
but only one batch size is included (configs for other batch sizes would be identical
except for the batch size).

=> We have possible matched, unmatched and matched+unmatched tables
    => Surrogate should use merged tables
    => Will just use matched tabular for now
=> Example config doesn't tell use which hyperparameters change or their boundaries,
   will have to use the json files and check which hyperparameters are constant and
   which are not.
   => For tabular setup with numerical values,
    * If evenly spaced ints, we can make it UniformInteger
    * Else, we are forced to make it Categorical.
        * If there is a natural ordering such as "depth = [1, 5, 16]" then this is
          Ordinal
        * Otherwise, Categorical

> The data were collected in two phases with _phase 0_ being a preliminary data
collection run and _phase 1_ designed to be nearly identical, but scaled up to more
points. It should be safe to combine the phases. However, for the ResNet50 on ImageNet
task, only phase 0 includes the 1024 batch size data since it was abandoned when
scaling up data collection in phase 1. In Wang et al., it is used for training but not
evaluation.

=> We only take phase1
=> For ResNet50_imagenet, we take the highest common one (not 1024)

> Each row corresponds to training a specific model on a specific dataset with one
particular setting of the hyperparameters (a.k.a. a "trial").
The "trial_dir" column should uniquely identify the trial.
Trials where training diverged are also included, but the "status" column should show
that they are "diverged."
That said, even if a trial has a normal status of "done" it might have been close to
diverging and have extremely high validation or training loss.

=> For tabular, I guess we just make these "divergent" values be np.inf/-np.inf

Files of interest after download, see README included to see contained items:
* pd1_matched_phase0_results.json
* pd1_matched_phase1_results.json
* pd1_unmatched_phase0_results.json
* pd1_unmatched_phase1_results.json

Datasets:
* translate_wmt
* uniref50
* lm1b
* svhn_no_extra
* imagenet
* mnist
* fashion_mnist
* cifar100
* cifar10

=> We need to create a seperate table for each of these
    * product(datasets, [matched, unmatched, mixed])

Metrics:
* valid/ce_loss
* valid/error_rate,
* train/ce_loss
* train/error_rate
* train_cost
* status <- "Maybe interesting, let's us know if something diverged"

**Later Edit:**

The datasets that make sense so that 1) no conditional HPS and 2) epochs and fidelities are equal are
- `f"{dataset_name}_{model}_{batch_size}"`


Conclusion
-----------
Create full mixture of valid tables we care about:

* ~Combine phase_0 and phase_1 json files~ Only use phase_1 data as it has phase_0 in it.
    * For ResNet50_imagenet, remove batch size entries 1024

* Extract seperate full tables for each `name, kind in product(datasets, [matched, unmatched])`
    - f"{name}_{kind}_all_budgets"

* Create merged tables
    - f"{name}_mixed_all_budgets"

* For all the above tables, extract only the max budget
    - f"{name}_{kind}_max_budget"
    - f"{name}_mixed_max_budget"
----
* For each of the `max_budget` tables, we now create the tabular benchmarks of interest,
  pruning out unused entries/constant entries to reduce table size. We also
  - f"tabular_{name}_{kind}_max_budget"
  - f"tabular_{name}_mixed_max_budget"

* As this is purely tabular and we need to encode it to a configspace, we used
  specifically only the "matched" variant with each dataset {name}. This means the hps
  are "evenly spaced" (Halton Sequence?)
    * A query that results in NaN values has to ensure we properly deal with how to
    encode the NaNs, probably with np.inf or -np.inf

---
* Later for surrogates, it makes sense to train them on the "mixed" variants as this has
  more data to use.
        * At this point, we also have to consider how to handle NaNs with the surrogate

### Trying out `phase_0` data

```python
# Split up the json files by datasets present
with open("pd1_matched_phase0_results.jsonl", "r") as f:
     x = json.load(f)
     
datasets = set([e["dataset"] for e in x])
d = { name: [] for name in datasets }
for e in x:
    d[e["dataset"]].append(e)

print(len(x))
2277

print(datasets)
{'svhn_no_extra', 'imagenet', 'translate_wmt', 'mnist', 'fashion_mnist', 'uniref50', 'cifar10', 'cifar100', 'lm1b'}
```

Each config for a dataset was evaluated at some epochs. Turns out this is not quite uniform
and so here we present the counts of how many configs have the same epochs
```python

s = { name: [] for name in d}
for name, entries in d.items():
    epochs = [(min(e["epoch"]), max(e["epoch"])) for e in entries if e["epoch"] is not None]
    s[name] = Counter(epochs)
pprint(s)

{'cifar10': Counter({(45, 299): 99,
                     (5, 299): 96,
                     (5, 22): 1,
                     (5, 45): 1,
                     (5, 11): 1}),
 'cifar100': Counter({(45, 199): 100, (5, 199): 99}),
 'fashion_mnist': Counter({(45, 199): 178,
                           (5, 199): 175,
                           (5, 249): 90,
                           (45, 249): 89,
                           (45, 45): 2,
                           (5, 5): 2,
                           (5, 11): 1,
                           (45, 91): 1,
                           (5, 17): 1}),
 'imagenet': Counter({(0, 99): 189, (0, 0): 3, (0, 20): 1}),
 'lm1b': Counter({(1, 74): 59, (1, 7): 1}),
 'mnist': Counter({(40, 199): 177,
                   (5, 199): 177,
                   (5, 249): 91,
                   (40, 249): 88,
                   (5, 10): 3,
                   (40, 40): 1,
                   (5, 5): 1}),
 'svhn_no_extra': Counter({(15, 199): 99, (3, 199): 99}),
 'translate_wmt': Counter({(0, 19): 68, (0, 1): 1, (0, 0): 1}),
 'uniref50': Counter({(0, 1): 72})}
```


### Trying out `phase_1` data
```python
# Same as above for loading and names
len(x)
9200
{'cifar10': Counter({(45, 299): 396,
                     (5, 299): 388,
                     (5, 17): 2,
                     (5, 5): 1,
                     (5, 11): 1,
                     (5, 22): 1,
                     (5, 39): 1,
                     (5, 28): 1}),
 'cifar100': Counter({(45, 199): 396,
                      (5, 199): 388,
                      (5, 5): 5,
                      (45, 45): 3,
                      (5, 11): 2,
                      (5, 45): 2,
                      (5, 28): 1}),
 'fashion_mnist': Counter({(45, 199): 681,
                           (5, 199): 681,
                           (45, 249): 363,
                           (5, 249): 337,
                           (5, 5): 5,
                           (5, 11): 4,
                           (45, 45): 1,
                           (5, 22): 1}),
 'imagenet': Counter({(0, 99): 749,
                      (0, 0): 17,
                      (0, 12): 1,
                      (0, 1): 1,
                      (0, 7): 1,
                      (0, 27): 1,
                      (0, 2): 1,
                      (0, 4): 1}),
 'lm1b': Counter({(1, 74): 251}),
 'mnist': Counter({(40, 199): 681,
                   (5, 199): 662,
                   (40, 249): 357,
                   (5, 249): 354,
                   (5, 5): 6,
                   (5, 10): 4,
                   (40, 40): 3,
                   (5, 25): 1,
                   (5, 15): 1}),
 'svhn_no_extra': Counter({(15, 199): 390,
                           (3, 199): 389,
                           (15, 15): 3,
                           (15, 30): 1,
                           (15, 77): 1,
                           (15, 46): 1,
                           (3, 3): 1,
                           (3, 30): 1,
                           (3, 7): 1,
                           (3, 23): 1,
                           (3, 27): 1,
                           (3, 11): 1}),
 'translate_wmt': Counter({(0, 19): 258, (0, 0): 3, (0, 1): 3}),
 'uniref50': Counter({(0, 1): 254})}
````

Seems that both are pretty consistent with the majority fidelitiy min max ranges.
We wil call these `ranges` where there are 1, 2 or 4 of them for each datasets, excluding the few that are outliers.

Notes: `"lm1b"` and  `"uniref50"` seems to only have one range while `"uniref50"` is only in `(0, 1)`.

### Epochs
We get the number of configs that ran to completion, failed and have no epochs, or those that have parial epochs

```python
for name, entries in d.items():
    ran = [e for e in entries if e["epoch"] is not None]
    failed = [e for e in entries if e["epoch"] is None]
    m = max([len(e["epoch"]) for e in ran])
    partial = [e for e in ran if len(e["epoch"]) < m/2]

    print(f"{name} r:{len(ran)} + f:{len(failed)} = {len(ran) + len(failed)} | partial:{len(partial)} | n_epoch_tiers:{m}")
    print("-----")
    s = set()
    for r in ran:
        s.add(tuple(r["epoch"]))
    l = sorted(list(s), key=lambda e: len(e))
    for ll in l:
        print(ll)
    print("-----")

# ... truncated output
# ...
# uniref50 r:254 + f:146 = 400 | partial:0 | n_epoch_tiers:22
# -----
# (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
# -----
# cifar10 r:791 + f:9 = 800 | partial:403 | n_epoch_tiers:53
# -----
# (5,)
# (5, 11)
# (5, 11, 17)
# (5, 11, 17, 22)
# (5, 11, 17, 22, 28)
# (45, 91, 136, 182, 227, 273, 299)
# (5, 11, 17, 22, 28, 34, 39)
# (5, 11, 17, 22, 28, 34, 39, 45, 51, 56, 62, 68, 73, 79, 85, 91, 96, 102, 108, 113, 119, 125, 130, 136, 142, 147, 153, 159, 164, 170, 176, 182, 187, 193, 199, 204, 210, 216, 221, 227, 233, 238, 244, 250, 256, 261, 267, 273, 278, 284, 290, 295, 299)
# -----
# cifar100 r:797 + f:3 = 800 | partial:409 | n_epoch_tiers:36
# -----
# (5,)
# (45,)
# (5, 11)
# (45, 91, 136, 182, 199)
# (5, 11, 17, 22, 28)
# (5, 11, 17, 22, 28, 34, 39, 45)
# (5, 11, 17, 22, 28, 34, 39, 45, 51, 56, 62, 68, 73, 79, 85, 91, 96, 102, 108, 113, 119, 125, 130, 136, 142, 147, 153, 159, 164, 170, 176, 182, 187, 193, 199, 199)
# -----
# lm1b r:251 + f:149 = 400 | partial:0 | n_epoch_tiers:38
# -----
# (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 74)
# -----
```
It seems that not all of them have a incremental +1 epoch range

### Hyperparameter ranges
We want to check the hyperparameter ranges for each dataset. Hopefully, we should be able to create a combinatorial grid from all of these.

```python
```

### Equality of configs
Seems we could care just about `entry["hparams"]`

Note: seems there are `lr_hparams` and `opt_hparams`, does this mean it's a conditional space?
