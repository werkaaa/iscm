import sys

sys.path.append('.')
import warnings

warnings.filterwarnings("ignore", message="No GPU automatically detected")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import argparse
from pathlib import Path
from collections import defaultdict
import copy
from tqdm import tqdm

import numpy as np

from experiment.utils import load_data, get_id, load_pred, check_env_var
from metrics.graph_distances import shd
from metrics.precission_recall import f1, precision, recall
from results_reproducibility.utils.parse import load_methods_config
from results_reproducibility.definitions import COV_PLOT, DISABLE_NOISE_PLOT, SCRATCH
from results_reproducibility.definitions import MOOIJ, SQUIRES
from visualization_paper.summary_plots import benchmark_summary, plot_experiment, plot_mean_cov, plot_induced_noise_variance_distribution
from visualization_paper.sortability_plot import sortability_plot, heuristic_sortability_plot
from data_generation.graph_utils import convert_to_binary_matrix


def make_summary(summary_path, result_paths, data_paths, experiment_name):
    """
    Args:
        summary_path
        result_paths (dict): {method: [paths to result csv's]}
        data_paths (list): [paths to data csv's]
        decision_threshold
        mask_definite_loops
        show_trivial

    """
    # id: data dict
    print("Generating summary before data")
    print(len(data_paths))
    # TODO: Reimplement so that we don't load all the data if not necessary
    data = {get_id(p): load_data(p) for p in data_paths}

    # compute data covariances
    dataset_cov = []
    dataset_info_list = []
    dataset_noise_vars = []
    dataset_prestand_vars = []
    print("Generating summary")

    is_heuristic = False
    for dataset in data.values():
        dataset_info_list.append(dataset["meta_info"])
        print(dataset["meta_info"], 'metainfo')
        if 'data_gen_method' in dataset["meta_info"].keys():
            is_heuristic = dataset["meta_info"]['data_gen_method'] in [MOOIJ, SQUIRES]

        # # Set the 'COV_PLOT' env variable to compute mean covariance.
        if check_env_var(COV_PLOT):
            dataset_cov.append(np.cov(dataset["sample"].T))
        if "noise_vars" in dataset.keys():
            dataset_noise_vars.append(dataset["noise_vars"])
        if "prestand_vars" in dataset.keys():
            dataset_prestand_vars.append(dataset["prestand_vars"])

    # # Set the 'COV_PLOT' env variable to compute mean covariance.
    if check_env_var(COV_PLOT):
        print('cov plot')
        plot_mean_cov(dataset_cov, dataset_info_list, summary_path)

    if len(dataset_noise_vars) > 0 and not check_env_var(DISABLE_NOISE_PLOT):
        plot_induced_noise_variance_distribution(dataset_prestand_vars, dataset_noise_vars, dataset_info_list, summary_path)

    # method: dict of metrics
    results = defaultdict(lambda: defaultdict(list))
    dataset_info = defaultdict(list)

    sortability_results = defaultdict(lambda: defaultdict(list))


    for method, mpaths in result_paths.items():

        print(f"Summarizing: {method}")

        # load predictions as id: prediction dict
        all_predictions = {get_id(p): load_pred(p) for p in mpaths}

        if len(mpaths) and np.isscalar(all_predictions[get_id(mpaths[0])]["predictions"]):
            print("Computing sortabilities")
            for pred_id, pred in tqdm(all_predictions.items()):
                keys = data[pred_id]["meta_info"].keys()
                for key in keys:
                    sortability_results[method][key].append(data[pred_id]["meta_info"][key])
                sortability_results[method]['val'].append(pred["predictions"])

        else:
            metrics_dict = {
                "shd": shd,
                "f1": f1,
                "precision": precision,
                "recall": recall
            }
            # compute metrics individually for every test case
            for pred_id, pred in all_predictions.items():
                assert pred_id in data.keys(), \
                    f"pred `{pred_id}` doesn't have matching data instance.\ndata_paths: {data_paths}\nmpaths: {mpaths}"

                # load ground truth graph
                g_true_bin = convert_to_binary_matrix(data[pred_id]["g"])
                g_pred_bin = convert_to_binary_matrix(pred["predictions"])

                # store dataset info and cov
                dataset_info[method].append(data[pred_id]["meta_info"])

                # compute metrics
                for metric_name, metric in metrics_dict.items():
                    results[method][metric_name].append(metric(g_true_bin, g_pred_bin))

            print("Computed metrics")
    if len(results) > 0:
        # dump all metrics for later plotting
        benchmark_summary(summary_path, copy.deepcopy(results), copy.deepcopy(dataset_info),
                          only_metrics=None, dump_main=True, dataset_info_skip=["num_samples", "graph_type"])

        # visualize preds
        plot_experiment(experiment_name, summary_path)

    if len(sortability_results) > 0:
        if is_heuristic:
            heuristic_sortability_plot(sortability_results, summary_path)
        else:
            sortability_plot(sortability_results, summary_path)

    print("Finished successfully.")


if __name__ == "__main__":
    """
    Runs plot call
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--methods_config_path", type=Path, required=True)
    parser.add_argument("--path_data", type=Path, required=True)
    parser.add_argument("--path_plots", type=Path, required=True)
    parser.add_argument("--path_results", type=Path, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--descr")
    kwargs = parser.parse_args()

    data_found = sorted([p for p in kwargs.path_data.iterdir() if p.is_dir()])

    if str(kwargs.path_results) != ".":
        methods_config_raw = load_methods_config(kwargs.methods_config_path, abspath=True)

        methods_config = {}
        for method in methods_config_raw.keys():
            methods_config[method] = methods_config_raw[method]

        results_p = sorted([p for p in kwargs.path_results.iterdir()])
        results_found = {}
        for meth, _ in methods_config.items():
            method_results = list(filter(lambda p: p.name.rsplit("_", 1)[0] == meth, results_p))
            results_found[meth] = method_results
    else:
        results_found = dict()

    make_summary(kwargs.path_plots, results_found, data_found, kwargs.experiment_name)
    print("Done.")
