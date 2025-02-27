from typing import Optional, Dict

import numpy as np

from data_generation.graph_utils import Graph


def sample_weight(a: float, b: float, rng: Optional[np.random.Generator] = None):
    """Samples weight uniformly from [-b, -a] U [a, b]"""
    assert a <= b
    if rng is None:
        rng = np.random.default_rng()
    x = rng.uniform(a - b, b - a)
    if x < 0:
        x -= a
    else:
        x += a
    return x


def linear_mechanism(node: int, all_generated: Dict, noise: np.array, graph: Graph):
    """Generates data for a single node according to linear mechanism."""
    generated = noise
    for parent in graph.get_parents(node):
        generated += graph.get_weight(parent, node) * all_generated[parent]

    return generated
