"""
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

Datasets X models X batch_size:
* translate_wmt X xformer
* uniref50 X Transformer
* lm1b X Transformer
* svhn_no_extra X WideResNet
* imagenet X ResNet50
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

Check investigation_data.md for more on these conclusions:
> 

Conclusion
-----------
Create full mixture of valid tables we care about:

* Combine phase_0 and phase_1 json files
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
"""
from pathlib import Path


def combine_json()


def process(datadir: Path) -> None:
    if not datadir.exists():
        raise FileNotFoundError(
            f"Can't find folder at {datadir}, have you run\n"
            f"`python -m mfpbench.download --data-dir {datadir.parent}`"
        )


if __name__ == "__main__":
    import argparse

    HERE = Path(__file__).resolve().absolute().parent
    DATADIR = HERE.parent.parent / "data" / "pd1-data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    datadir = args.data_dir if args.data_dir else DATADIR
    process(datadir)
