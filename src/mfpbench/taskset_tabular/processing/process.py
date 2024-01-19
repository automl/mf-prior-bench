'''
Download and basic processing code source:
https://github.com/releaunifreiburg/DPL/blob/main/python_scripts/download_task_set_data.py
'''

import argparse
import json
import os
import urllib
from concurrent import futures
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow.compat.v2 as tf  # for gfile.
import time
import tqdm


PAPER_TASK_FAMILIES = [
    "mlp",
    "mlp_ae",
    "mlp_vae",
    "conv_pooling",
    "conv_fc",
    "nvp",
    "maf",
    "char_rnn_language_model",
    "word_rnn_language_model",
    "rnn_text_classification",
    "quadratic",
    "losg_tasks",
]

# source: https://github.com/releaunifreiburg/DPL/tree/main/data/taskset
DPL_TASK_FAMILIES = [
    "FixedTextRNNClassification_imdb_patch32_GRU64_avg_bs128",
    "FixedTextRNNClassification_imdb_patch32_GRU128_bs128",
    "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_avg_bs128",
    "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_last_bs128",
    "FixedTextRNNClassification_imdb_patch32_LSTM128_bs128",
    "FixedTextRNNClassification_imdb_patch32_LSTM128_E128_bs128",
    "FixedTextRNNClassification_imdb_patch32_VRNN64_relu_avg_bs128",
    "FixedTextRNNClassification_imdb_patch32_VRNN64_tanh_avg_bs128",
    "FixedTextRNNClassification_imdb_patch32_VRNN128_tanh_bs128",
    "FixedTextRNNClassification_imdb_patch128_LSTM128_avg_bs64",
    "FixedTextRNNClassification_imdb_patch128_LSTM128_bs64",
    "FixedTextRNNClassification_imdb_patch128_LSTM128_embed128_bs6",
]

OPTIMIZERS_TO_CONSIDER = [
    "adam8p_wide_grid_1k",
    # "adam6p_wide_grid_1k",
    "adam4p_wide_grid_1k",
    # "adam1p_wide_grid_1k",
    # "nadamw",
]

HP_BASE_URL = "https://raw.githubusercontent.com/google-research/google-research/master/task_set/optimizers/configs/"
HP_URL = lambda x: HP_BASE_URL + f"{x}.json"


gfile = tf.io.gfile


def load_joint_cache(task, opt_set_name):
    """Loads the learning curves for the given task and opt_set_name."""
    base_dir = "gs://gresearch/task_set_data/"
    p = os.path.join(
        base_dir, task, f"{opt_set_name}_10000_replica5.npz"
    )
    cc = np.load(gfile.GFile(p, "rb"))
    return cc["optimizers"], cc["xs"], cc["ys"]


def threaded_tqdm_map(threads, func, data):
    """Helper that does a map on multiple threads."""
    future_list = []
    with futures.ThreadPoolExecutor(threads) as executor:
        for l in tqdm.tqdm(data, position=0):
            future_list.append(executor.submit(func, l))
        return [x.result() for x in tqdm.tqdm(future_list, position=0)]


def load_tasks(tasks):
    """Multi threaded loading of all data for each task.
    Args:
      tasks: list of task names
    Returns:
      A dictionary mapping taks name to tuples of:
      (optimizer names, x data points, and y data points)
    """

    def load_one(t):
        adam8p = load_joint_cache(t, "adam8p_wide_grid_1k")
        adam6p = load_joint_cache(t, "adam6p_wide_grid_1k")
        adam4p = load_joint_cache(t, "adam4p_wide_grid_1k")
        adam1p = load_joint_cache(t, "adam1p_wide_grid_1k")
        nadamw = load_joint_cache(t, "nadamw_grid_1k")
        return {
            "adam8p_wide_grid_1k": adam8p,
            "adam6p_wide_grid_1k": adam6p,
            "adam4p_wide_grid_1k": adam4p,
            "adam1p_wide_grid_1k": adam1p,
            "nadamw": nadamw,
        }

    results = threaded_tqdm_map(100, load_one, tasks)

    for k, v in zip(tasks, results):
        if v is None:
            print("No data found for task: %s" % k)

    return {k: v for k, v in zip(tasks, results) if v is not None}


def get_task_names():
    content = gfile.GFile("gs://gresearch/task_set_data/task_names.txt").read()
    return sorted(content.strip().split("\n"))


def get_hyperparameter_list(optimizer: str, optimizer_names: list) -> pd.DataFrame:
    path = HP_URL(optimizer.strip("_1k"))
    try:
        configs = json.loads(urllib.request.urlopen(path).read())
    except Exception as e:
        print(f"Failed for {optimizer}")
        return None
    hparam_dicts = {
        i: configs[optname.decode("utf8")][0] 
        for i, optname in enumerate(optimizer_names, start=1)
    }
    hp_df = pd.DataFrame.from_dict(hparam_dicts, orient="index")
    hp_df = hp_df.reset_index().rename(columns={"index": "id"})

    return hp_df


def get_curves(x: np.ndarray, y: np.ndarray):
    nr_seeds = y.shape[1]
    nr_optimizations = y.shape[0]
    train_curves = []
    val_curves = []
    test_curves = []

    for hp_index in range(nr_optimizations):

        train_seed_curve = []
        valid_seed_curve = []
        test_seed_curve = []

        for seed_index in range(nr_seeds):
            train_seed_curve.append(y[hp_index, seed_index, :, 0])
            valid_seed_curve.append(y[hp_index, seed_index, :, 1])
            test_seed_curve.append(y[hp_index, seed_index, :, 2])

        train_seed_curve = np.mean(train_seed_curve, axis=0)
        valid_seed_curve = np.mean(valid_seed_curve, axis=0)
        test_seed_curve = np.mean(test_seed_curve, axis=0)

        train_curves.append(train_seed_curve.tolist())
        val_curves.append(valid_seed_curve.tolist())
        test_curves.append(test_seed_curve.tolist())

    return train_curves, val_curves, test_curves


def collate_data_to_table(
    hp_df: pd.DataFrame, train_curves: list, val_curves: list, test_curves: list
) -> pd.DataFrame:
    df = pd.DataFrame()
    for hp_index, hp_config in enumerate(hp_df.id.values):
        assert len(train_curves[hp_index]) == len(val_curves[hp_index]), "Num. steps do not match!"
        assert len(test_curves[hp_index]) == len(val_curves[hp_index]), "Num. steps do not match!"
        hp_config_result = {
            "id": [hp_config] * len(train_curves[hp_index]),
            # `loss`, `error`, `cost` are mfpbench Result attributes, names cannot be same
            "train_loss": train_curves[hp_index],
            "valid_loss": val_curves[hp_index],
            "test_loss": test_curves[hp_index],
            "epoch": np.arange(1, len(val_curves[hp_index]) + 1),
            # TODO: be more elegant with cost?
            # Right now, placeholder mainly for mfpbench API as TaskSet has no cost info
            "train_cost": np.arange(1, len(val_curves[hp_index]) + 1)
        }
        _df = pd.DataFrame.from_dict(hp_config_result)
        df = pd.concat((df, _df))
    # adding hyperparameters to the table
    df = pd.merge(df, hp_df, on="id", how="left")
    # resetting to MultiIndex
    df = df.set_index(["id", "epoch"])

    return df


def process_taskset(task_family: str, output_dir: Path) -> None:
    # task_id = args.task_id
    task_names = get_task_names()  # returns a list of str
    tasks = pd.Series(task_names)

    TASK_LIST = DPL_TASK_FAMILIES if task_family == "dpl" else PAPER_TASK_FAMILIES
    # filter for concerned tasks
    tasks = tasks.loc[
        tasks.apply(lambda x: any(x.startswith(_s) for _s in TASK_LIST))
    ]

    if task_family == "dpl":
        df = tasks
    else:
        # grouping
        _task_col = tasks.apply(lambda x: x.split("_seed")[0])
        _seed_col = tasks.apply(lambda x: x.split("_seed")[-1])
        df = pd.DataFrame({"task": _task_col, "seed": _seed_col})

        def full_task_name(task, seed):
            return f"{task}_seed{seed}"

    output_dir.mkdir(parents=True, exist_ok=True)

    time_taken = 0
    for idx in df.index.values:
        start = time.time()
        if task_family == "dpl":
            task_name = df.loc[idx]
        else:
            task_name = full_task_name(*df.loc[idx].values)
        results = load_tasks([task_name])
        optimizer_families = [
            _opt for _opt in results[task_name].keys() if _opt in OPTIMIZERS_TO_CONSIDER
        ]
        results = load_tasks([task_name])
        for optimizer_name in optimizer_families:
            _optimizer_names, x, y = results[task_name][optimizer_name]
            hp_df = get_hyperparameter_list(optimizer_name, _optimizer_names)
            if hp_df is None:
                continue
            train_curves, valid_curves, test_curves = get_curves(x, y) 
            table = collate_data_to_table(hp_df, train_curves, valid_curves, test_curves)
            if task_family != "dpl":
                # shortening name
                task_name = task_name.replace(
                    "FixedTextRNNClassification_imdb_patch128_LSTM128", "taskset_nlp"
                )
            filename = f"{task_name}_{optimizer_name.split('_')[0]}.parquet"
            table.to_parquet(output_dir / filename)
        end = time.time()
        print(f"Time taken for {task_name}: {(end-start):.2f}s")
        time_taken += (end - start)
    print(f"Total time taken: {(time_taken):.2f}s")
    return


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Prepare hyperparameter candidates from the taskset task',
    )
    parser.add_argument(
        '--task_family',
        help='The task family to consider',
        type=str,
        default="dpl",
    )
    parser.add_argument(
        '--output_dir',
        help='The output directory where the validation curves and hyperparameter configurations will be saved',
        type=str,
        default='./taskset',
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = get_arguments()

    process_taskset(args.task_set, args.output_dir)
# end of file