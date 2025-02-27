from typing import Dict, Optional
from collections import defaultdict

import numpy as np

from data_generation.graph_utils import Graph, topo_sort


def get_exact_covariance_for_alternate_standardization(graph: Graph, noise_vars: Dict[int, float]):
    sorted_nodes = topo_sort(graph)
    cov = np.eye(graph.get_num_nodes())
    weights = graph.get_matrix_representation()
    weights_corrected = np.zeros_like(weights)
    noise_vars_corrected = defaultdict()
    for i, end_node in enumerate(sorted_nodes):
        parents_weights = weights[:, end_node]
        correction = parents_weights.T @ cov @ parents_weights + noise_vars[end_node]
        weights_corrected[:, end_node] = parents_weights / np.sqrt(correction)
        noise_vars_corrected[end_node] = noise_vars[end_node] / correction

        for start_node in sorted_nodes[:i]:
            cov[start_node, end_node] = weights_corrected[:, end_node].T @ cov[start_node, :]
            cov[end_node, start_node] = cov[start_node, end_node]
    return cov, weights_corrected, noise_vars_corrected


def get_data_samples_from_noise_vars(noise_vars: Dict[int, float], weights: np.array, num_samples: int,
                                     rng: np.random.Generator):
    noise_vars_list = [noise_vars[k] for k in range(len(noise_vars))]
    noise_samples = [rng.normal(loc=np.zeros_like(noise_vars_list), scale=np.sqrt(noise_vars_list)) for _ in
                     range(num_samples)]
    N = np.array(noise_samples).T
    X = np.linalg.inv(np.eye(len(noise_vars)) - weights.T) @ N
    return X


def generate_linear_data_raw(graph: Graph, noise_vars: Dict[int, float], num_samples: int,
                             rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random.default_rng()
    weights = graph.get_matrix_representation()
    X = get_data_samples_from_noise_vars(noise_vars, weights, num_samples, rng)
    return X.T


def generate_linear_data_alternate_standardization(graph: Graph, noise_vars: Dict[int, float], num_samples: int,
                                                   rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random.default_rng()
    _, weights_corrected, noise_vars = get_exact_covariance_for_alternate_standardization(graph, noise_vars)
    X = get_data_samples_from_noise_vars(noise_vars, weights_corrected, num_samples, rng)
    return X.T
