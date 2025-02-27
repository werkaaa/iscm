import itertools
from collections import defaultdict
from types import SimpleNamespace

from results_reproducibility.utils.yaml_loader import load_config


def cartesian_dict(d):
    """
    Cartesian product of nested dict/defaultdicts of lists
    Example:

    d = {'s1': {'a': 0,
                'b': [0, 1, 2]},
         's2': {'c': [0, 1],
                'd': [0, 1]}}

    yields
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 0, 'd': 0}}
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 0, 'd': 1}}
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 1, 'd': 0}}
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 1, 'd': 1}}
        {'s1': {'a': 0, 'b': 1}, 's2': {'c': 0, 'd': 0}}
        ...
    """
    if type(d) in [dict, defaultdict]:
        keys, values = d.keys(), d.values()
        for c in itertools.product(*(cartesian_dict(v) for v in values)):
            yield dict(zip(keys, c))
    elif type(d) == list:
        for c in d:
            yield from cartesian_dict(c)
    elif type(d) == str and ',' in d:
        res = []
        for elem in d.split(','):
            res.append(float(elem))
        yield tuple(res)
    else:
        yield d


def load_data_config(path, abspath=False, verbose=False):
    """Load yaml config for data specification"""

    config = load_config(path, abspath=abspath)
    if config is None:
        return None

    spec = {"data": {}}

    # add meta info (all but "data")
    for key, val in config.items():
        if key not in spec:
            spec[key] = val

    # process data field
    spec["data"] = list(cartesian_dict(config["data"]))

    return SimpleNamespace(**spec)


def load_methods_config(path, abspath=False, verbose=False):
    """Load yaml config for method specification"""

    config = load_config(path, abspath=abspath)
    if config is None:
        return None
    return config
