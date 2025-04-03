from typing import Optional

import numpy as np

from iscm.graph_utils import Graph, topo_sort

def generate_data_squires(graph: Graph, num_samples: int, rng: Optional[np.random.Generator] = None):
    """Implemented as in https://proceedings.mlr.press/v177/squires22a/squires22a.pdf."""
    all_generated = {node: None for node in graph.get_nodes()}
    sorted_nodes = topo_sort(graph)
    variances = {node: None for node in graph.get_nodes()}
    if rng is None:
        rng = np.random.default_rng()
    for node in sorted_nodes:
        noise_var = 0.5 if len(graph.get_parents(node)) > 0 else 1.0
        noise_contribution = rng.normal(loc=0, scale=np.sqrt(noise_var), size=num_samples)
        parents_contribution = np.zeros_like(noise_contribution)
        for parent in graph.get_parents(node):
            parents_contribution += graph.get_weight(parent, node) * all_generated[parent]

        if len(graph.get_parents(node)) > 0:
            generated = parents_contribution / (np.sqrt(2) * np.std(parents_contribution, ddof=1)) + noise_contribution
        else:
            generated = noise_contribution

        all_generated[node] = generated
        variances[node] = np.var(generated, ddof=1)

    return np.array([all_generated[node] for node in graph.get_nodes()]).T


def generate_data_mooij(graph: Graph, num_samples: int, scale_noise: bool, rng: Optional[np.random.Generator] = None):
    all_generated = {node: None for node in graph.get_nodes()}
    sorted_nodes = topo_sort(graph)
    variances = {node: None for node in graph.get_nodes()}
    if rng is None:
        rng = np.random.default_rng()
    for node in sorted_nodes:
        noise_var = 1.0
        scaling = 1.0
        for parent in graph.get_parents(node):
            scaling += graph.get_weight(parent, node) ** 2
        scaling = np.sqrt(scaling)

        noise_contribution = rng.normal(loc=0, scale=np.sqrt(noise_var), size=num_samples)
        parents_contribution = np.zeros_like(noise_contribution)
        for parent in graph.get_parents(node):
            parents_contribution += graph.get_weight(parent, node) / scaling * all_generated[parent]

        if scale_noise:
            noise_contribution /= scaling

        generated = parents_contribution + noise_contribution
        all_generated[node] = generated
        variances[node] = np.var(generated, ddof=1)

    return np.array([all_generated[node] for node in graph.get_nodes()]).T
