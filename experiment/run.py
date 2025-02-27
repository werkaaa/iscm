import sys

sys.path.append('.')
import argparse
import re
import platform
from pathlib import Path
import json
import os
import subprocess

from experiment.utils import load_data, timer, NumpyJSONEncoder
from experiment.methods import *
from results_reproducibility.utils.parse import load_methods_config

from results_reproducibility.definitions import NOTEARS, AVICI, VAR_SORT_REGRESS, R2_SORT_REGRESS, RANDOM_SORT_REGRESS, \
    R2_SORTABILITY, CEV_SORTABILITY, INDG_SORTABILITY, NS_SORTABILITY, MBS_SORTABILITY, VAR_SORTABILITY, PC, GES, GOLEM_EV, GOLEM_NV, \
    LINGAM, SCORE, CAM

if __name__ == "__main__":
    """
    Runs methods on a data instance and creates predictions 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--descr", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--path_results", type=Path, required=True)
    parser.add_argument("--path_data_root", type=Path)
    parser.add_argument("--data_id", type=int)
    parser.add_argument("--path_methods_config", type=Path, required=True)
    kwargs = parser.parse_args()

    # generate directory if it doesn't exist
    kwargs.path_results.mkdir(exist_ok=True, parents=True)
    (kwargs.path_results / "logs").mkdir(exist_ok=True, parents=True)

    # get cpu info
    cpu_model = "not detected"
    if platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                cpu_model = re.sub(".*model name.*:", "", line, 1)

    # load data
    assert kwargs.data_id is not None
    methods_config = load_methods_config(kwargs.path_methods_config, abspath=True)
    assert kwargs.method in methods_config, f"{kwargs.method} not in config with keys {list(methods_config.keys())}"
    config = methods_config[kwargs.method]

    data = load_data(kwargs.path_data_root / f"{kwargs.data_id}")

    # run method and measure walltime
    ps = []
    base_run_name = f"{kwargs.method}_{kwargs.data_id}.json"

    if not os.path.isfile(kwargs.path_results / base_run_name):
        base_method = kwargs.method.split("__")[0]  # catch hyperparameter calibration case where name differs

        with timer() as walltime:

            # causal structure discovery
            if base_method == NOTEARS:
                pred = run_notears(data, config)
            elif base_method == AVICI:
                pred = run_avici(data, config)
            elif base_method == VAR_SORT_REGRESS:
                pred = run_var_sort_regress(data, config)
            elif base_method == R2_SORT_REGRESS:
                pred = run_r2_sort_regress(data, config)
            elif base_method == RANDOM_SORT_REGRESS:
                pred = run_random_sort_regress(data, config, seed=kwargs.data_id)
            elif base_method == PC:
                pred = run_pc(data, config, seed=kwargs.data_id)
            elif base_method == GES:
                pred = run_ges(data, config, seed=kwargs.data_id)
            elif base_method == GOLEM_EV:
                pred = run_golem(data, config, seed=kwargs.data_id)
            elif base_method == GOLEM_NV:
                pred = run_golem(data, config, seed=kwargs.data_id)
            elif base_method == LINGAM:
                pred = run_lingam(data, config, seed=kwargs.data_id)
            elif base_method == SCORE:
                pred = run_score(data, config)
            elif base_method == CAM:
                pred = run_cam(data, config)
            # sortabilities
            elif base_method == R2_SORTABILITY:
                pred = run_r2_sortability(data, config)
            elif base_method == VAR_SORTABILITY:
                pred = run_var_sortability(data, config)
            elif base_method == CEV_SORTABILITY:
                pred = run_cev_sortability(data, config)
            elif base_method == INDG_SORTABILITY:
                pred = run_indg_sortability(data, config)
            elif base_method == NS_SORTABILITY:
                pred = run_neighbour_count_sortability(data, config)
            elif base_method == MBS_SORTABILITY:
                pred = run_markov_blanket_size_sortability(data, config)

            else:
                raise KeyError(f"Unknown method `{kwargs.method}`")

            ps.append((base_run_name, pred))

        t_finish = walltime() / 60.0  # mins
        """Save predictions"""
        for run_name, p in ps:
            p["config"] = config
            p["cpu_model"] = cpu_model
            p["walltime"] = t_finish

            with open(kwargs.path_results / run_name, "w") as file:
                json.dump(p, file, indent=4, sort_keys=True, cls=NumpyJSONEncoder)

        print(f"{kwargs.descr}:  {kwargs.method} data_id {kwargs.data_id} finished successfully.")
