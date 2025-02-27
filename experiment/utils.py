import json
import os
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from results_reproducibility.definitions import FILE_DATA_META, FILE_DATA_G, FILE_DATA_SAMPLES, FILE_DATA_NOISE_VARS, \
    FILE_DATA_PRESTAND_VARS


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NumpyJSONDecoder(json.JSONDecoder):
    def _postprocess(self, obj):
        if isinstance(obj, dict):
            return {k: self._postprocess(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return np.array([self._postprocess(v) for v in obj])
        else:
            return obj

    def decode(self, obj, recurse=False):
        decoded = json.JSONDecoder.decode(self, obj)
        return self._postprocess(decoded)


@contextmanager
def timer() -> float:
    start = time.time()
    yield lambda: time.time() - start


def get_id(path):
    return int(path.name.split("_")[-1].split('.')[0])


def save_csv(arr, path):
    if arr.size > 0:
        pd.DataFrame(arr).to_csv(
            path,
            index=False,
            header=False,
            float_format="%.8f",
        )


def load_data(path, just_meta=False):
    loaded_data = {}

    # load meta data
    with open(path / FILE_DATA_META, "r") as file:
        loaded_data["meta_info"] = json.load(file)

    if just_meta:
        return loaded_data

    # load g
    try:
        loaded_data["g"] = np.array(pd.read_csv(path / FILE_DATA_G, index_col=False, header=None), dtype=np.float64)
    except:
        loaded_data["g"] = np.array([])

    # load data
    p = path / FILE_DATA_SAMPLES
    if p.is_file():
        loaded_data["sample"] = np.array(pd.read_csv(p, index_col=False, header=None), dtype=np.float64)

    # load variances if present
    p = path / FILE_DATA_NOISE_VARS
    if p.is_file():
        loaded_data["noise_vars"] = np.array(pd.read_csv(p, index_col=False, header=None), dtype=np.float64).flatten()

    p = path / FILE_DATA_PRESTAND_VARS
    if p.is_file():
        loaded_data["prestand_vars"] = np.array(pd.read_csv(p, index_col=False, header=None),
                                                dtype=np.float64).flatten()

    return loaded_data


def load_pred(path):
    print(path)
    with open(path, "r") as file:
        pred = json.load(file, cls=NumpyJSONDecoder)
        return pred


def check_env_var(env_var):
    return os.getenv(env_var, 'False').lower() == 'true'

