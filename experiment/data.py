import sys

sys.path.append('.')
from types import SimpleNamespace

import numpy as np
import argparse
import json
from pathlib import Path

from data_generation.data_gen_for_experiments import generate
from experiment.utils import save_csv
from results_reproducibility.utils.parse import load_data_config

from results_reproducibility.definitions import RNG_ENTROPY_EVAL, RNG_ENTROPY_HPARAMS, FILE_DATA_META, FILE_DATA_G, \
    FILE_DATA_SAMPLES, EVAL_MODE, HPARAMS_MODE, FILE_DATA_PRESTAND_VARS, FILE_DATA_NOISE_VARS

if __name__ == "__main__":
    """
    Generates data for experiment
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--descr", type=str, required=True)
    parser.add_argument("--data_config_path", type=Path, required=True)
    parser.add_argument("--path_data", type=Path, required=True)
    parser.add_argument("--j", type=int, required=True)
    kwargs = parser.parse_args()
    data_config = load_data_config(kwargs.data_config_path, verbose=False)

    if data_config.mode == HPARAMS_MODE:
        rng = np.random.default_rng(np.random.SeedSequence(entropy=(RNG_ENTROPY_HPARAMS, kwargs.j)))
    elif data_config.mode == EVAL_MODE:
        rng = np.random.default_rng(np.random.SeedSequence(entropy=(RNG_ENTROPY_EVAL, kwargs.j)))
    else:
        raise NotImplementedError(f"Data mode {data_config.mode} not known")

    spec_list = data_config.data

    spec = spec_list[(kwargs.j - 1) % len(spec_list)]
    spec["id"] = kwargs.j
    spec["data_folder"] = str(kwargs.path_data)
    spec = SimpleNamespace(**spec)

    if 'rff' in vars(data_config) and data_config.rff:
        rff = True
    else:
        rff = False

    if 'heuristic' in vars(data_config) and data_config.heuristic:
        print("generating heuristic")
        heuristic = data_config.heuristic
    else:
        heuristic = None

    if 'save_vars' in vars(data_config) and data_config.save_vars:
        save_vars = True
    else:
        save_vars = False

    if 'skip_save_data_and_graph' in vars(data_config) and data_config.skip_save_data_and_graph:
        skip_save_data_and_graph = True
    else:
        skip_save_data_and_graph = False

    if 'induced' in vars(data_config) and data_config.induced:
        induced = True
    else:
        induced = False

    if 'noise_dist' in vars(data_config) and data_config.noise_dist:
        noise_dist = data_config.noise_dist
    else:
        noise_dist = None

    data = generate(
        rng,
        num_samples=data_config.num_samples,
        graph_type=data_config.graph_type,
        spec=spec,
        rff=rff,
        save_vars=save_vars,
        induced=induced,
        heuristic=heuristic,
        noise_dist_config=noise_dist
    )

    if skip_save_data_and_graph:
        data["graph"] = np.array([])
        data["sample"] = np.array([])

    # write to file
    data_folder = kwargs.path_data / f"{kwargs.j}"
    data_folder.mkdir(exist_ok=True, parents=True)

    save_csv(data["graph"], data_folder / FILE_DATA_G)
    save_csv(data["sample"], data_folder / FILE_DATA_SAMPLES)
    save_csv(data["noise_vars"], data_folder / FILE_DATA_NOISE_VARS)

    if "prestand_vars" in data.keys():
        save_csv(data["prestand_vars"], data_folder / FILE_DATA_PRESTAND_VARS)

    meta_info_path = data_folder / FILE_DATA_META

    with open(meta_info_path, "w") as file:
        meta_info = {
            "num_samples": data_config.num_samples,
            "graph_type": data_config.graph_type,
            **vars(spec)
        }
        json.dump(meta_info, file, indent=4, sort_keys=True)

    print(f"{kwargs.descr}: {kwargs.j} finished successfully.")
