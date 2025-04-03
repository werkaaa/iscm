from inspect import currentframe, getframeinfo
from typing import Dict, Optional, Callable, Tuple

import numpy as np

from iscm.graph_utils import Graph, topo_sort
from iscm.data_genertaion_utils import check_node_number
from iscm.mechanisms.linear import linear_mechanism
from iscm.mechanisms.rff import rff_mechanism, draw_rff_params

GAUSSIAN = 'gaussian'
UNIFORM = 'uniform'


# GENERAL
###################################################################


def generate_node_data(node, all_generated, graph: Graph, noise_vars: Dict[int, float], num_samples: int,
                       mechanism: Callable,
                       rng: Optional[np.random.Generator], noise_dist: Optional[str] = None):
    """Generates single node data according to given mechanism."""

    if noise_dist == UNIFORM:
        noise = rng.uniform(
            low=-1 * 3 ** 0.5 * np.sqrt(noise_vars[node]),
            high=1 * 3 ** 0.5 * np.sqrt(noise_vars[node]),
            size=num_samples)
    else:
        noise = rng.normal(loc=0, scale=np.sqrt(noise_vars[node]), size=num_samples)

    if len(graph.get_parents(node)) > 0:
        generated = mechanism(node, all_generated, noise, graph)
    else:
        generated = noise

    return generated


def generate_data_raw(graph: Graph, noise_vars: Dict[int, float], num_samples: int, mechanism: Callable,
                      rng: Optional[np.random.Generator] = None, noise_dist=GAUSSIAN):
    """Generates data according to an SCM."""

    info = getframeinfo(currentframe())
    check_node_number(graph, noise_vars, f'{info.filename}::{info.lineno}')

    if rng is None:
        rng = np.random.default_rng()

    all_generated = {node: None for node in graph.get_nodes()}
    sorted_nodes = topo_sort(graph)

    for node in sorted_nodes:
        all_generated[node] = generate_node_data(
            node,
            all_generated,
            graph,
            noise_vars,
            num_samples,
            mechanism,
            rng,
            noise_dist=noise_dist,
        )

    X = np.array([all_generated[node] for node in graph.get_nodes()])
    return X.T


def generate_data_final_standardization(graph: Graph, noise_vars: Dict[int, float], num_samples: int,
                                        mechanism: Callable, rng: Optional[np.random.Generator] = None,
                                        return_prestand_variances: bool = False, noise_dist=GAUSSIAN):
    """Generates data according to a standardized SCM."""

    X = generate_data_raw(graph, noise_vars, num_samples, mechanism, rng, noise_dist=noise_dist)
    X_standardized = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True, ddof=1)
    return (X_standardized, np.var(X, axis=0, ddof=1)) if return_prestand_variances else X_standardized


def generate_data_alternate_standardization(graph: Graph, noise_vars: Dict[int, float], num_samples: int,
                                            mechanism: Callable, rng: Optional[np.random.Generator] = None,
                                            return_prestand_variances: bool = False, noise_dist=GAUSSIAN):
    """Generates data according to iSCM."""

    info = getframeinfo(currentframe())
    check_node_number(graph, noise_vars, f'{info.filename}::{info.lineno}')

    if rng is None:
        rng = np.random.default_rng()

    all_generated = {node: None for node in graph.get_nodes()}
    variances_dict = {node: None for node in graph.get_nodes()}
    sorted_nodes = topo_sort(graph)

    for node in sorted_nodes:
        generated = generate_node_data(
            node,
            all_generated,
            graph,
            noise_vars,
            num_samples,
            mechanism,
            rng,
            noise_dist=noise_dist
        )
        all_generated[node] = (generated - np.mean(generated)) / np.std(generated, ddof=1)
        variances_dict[node] = np.var(generated, ddof=1)

    X = np.array([all_generated[node] for node in graph.get_nodes()])
    variances = np.array([variances_dict[node] for node in graph.get_nodes()])
    return (X.T, variances) if return_prestand_variances else X.T


# SPECIFIC MECHANISMS
###################################################################
def generate_linear_data_raw(graph: Graph, noise_vars: Dict[int, float], num_samples: int,
                             rng: Optional[np.random.Generator] = None, noise_dist=GAUSSIAN):
    return generate_data_raw(graph=graph, noise_vars=noise_vars, num_samples=num_samples, mechanism=linear_mechanism,
                             rng=rng, noise_dist=noise_dist)


def generate_linear_data_alternate_standardization(graph: Graph, noise_vars: Dict[int, float], num_samples: int,
                                                   rng: Optional[np.random.Generator] = None,
                                                   return_prestand_variances: bool = False, noise_dist=GAUSSIAN):
    return generate_data_alternate_standardization(graph=graph, noise_vars=noise_vars, num_samples=num_samples,
                                                   mechanism=linear_mechanism, rng=rng,
                                                   return_prestand_variances=return_prestand_variances,
                                                   noise_dist=noise_dist)


def generate_linear_data_final_standardization(graph: Graph, noise_vars: Dict[int, float], num_samples: int,
                                               rng: Optional[np.random.Generator] = None,
                                               return_prestand_variances: bool = False, noise_dist=GAUSSIAN):
    return generate_data_final_standardization(graph=graph, noise_vars=noise_vars, num_samples=num_samples,
                                               mechanism=linear_mechanism, rng=rng,
                                               return_prestand_variances=return_prestand_variances,
                                               noise_dist=noise_dist)


def generate_rff_data_raw(graph: Graph, noise_vars: Dict[int, float], num_samples: int,
                          length_scale_range: Tuple[float], output_scale_range: Tuple[float],
                          rng: Optional[np.random.Generator] = None, noise_dist=GAUSSIAN):
    def mechanism(node, all_generated, noise, g):
        return rff_mechanism(
            node, all_generated, noise, g,
            **draw_rff_params(rng, d=len(g.get_parents(node)),
                              length_scale_range=length_scale_range,
                              output_scale_range=output_scale_range))

    return generate_data_raw(
        graph=graph,
        noise_vars=noise_vars,
        num_samples=num_samples,
        mechanism=mechanism,
        rng=rng,
        noise_dist=noise_dist
    )


def generate_rff_data_alternate_standardization(graph: Graph, noise_vars: Dict[int, float], num_samples: int,
                                                length_scale_range: Tuple[float],
                                                output_scale_range: Tuple[float],
                                                rng: Optional[np.random.Generator] = None, noise_dist=GAUSSIAN):
    if rng is None:
        rng = np.random.default_rng()

    def mechanism(node, all_generated, noise, g):
        return rff_mechanism(
            node, all_generated, noise, g,
            **draw_rff_params(rng, d=len(g.get_parents(node)),
                              length_scale_range=length_scale_range,
                              output_scale_range=output_scale_range))

    return generate_data_alternate_standardization(graph=graph, noise_vars=noise_vars, num_samples=num_samples,
                                                   mechanism=mechanism, rng=rng, noise_dist=noise_dist)


def generate_rff_data_final_standardization(graph: Graph, noise_vars: Dict[int, float], num_samples: int,
                                            length_scale_range: Tuple[float],
                                            output_scale_range: Tuple[float],
                                            rng: Optional[np.random.Generator] = None, noise_dist=GAUSSIAN):
    if rng is None:
        rng = np.random.default_rng()

    def mechanism(node, all_generated, noise, g):
        return rff_mechanism(
            node, all_generated, noise, g,
            **draw_rff_params(rng, d=len(g.get_parents(node)),
                              length_scale_range=length_scale_range,
                              output_scale_range=output_scale_range))

    return generate_data_final_standardization(graph=graph, noise_vars=noise_vars, num_samples=num_samples,
                                               mechanism=mechanism, rng=rng, noise_dist=noise_dist)


# NOISE TRANSFER
###################################################################

def generate_linear_data_with_transferred_noise(
        graph: Graph, noise_vars: Dict[int, float],
        induced_noise_vars: Dict[int, float],
        num_samples: int,
        rng: Optional[np.random.Generator] = None):
    """Generates data from a system with induced noise variances but keeping the marginal variance unchanged."""

    data_raw = generate_linear_data_raw(graph=graph, noise_vars=noise_vars, num_samples=num_samples, rng=rng)
    real_vars = np.var(data_raw, axis=0, ddof=1)
    scaling = np.sqrt(np.maximum(real_vars - induced_noise_vars, 0))

    all_generated = {node: None for node in graph.get_nodes()}
    sorted_nodes = topo_sort(graph)

    for node in sorted_nodes:

        generated = 0
        for parent in graph.get_parents(node):
            generated += graph.get_weight(parent, node) * all_generated[parent]

        # We are using the induced noise variances!
        # Note that the noise is added as it is, so this is the true noise variance.
        noise = rng.normal(loc=0, scale=np.sqrt(induced_noise_vars[node]), size=num_samples)
        if isinstance(generated, np.ndarray):
            all_generated[node] = scaling[node] * generated / np.std(generated, ddof=1) + noise
        else:
            all_generated[node] = noise

    X = np.array([all_generated[node] for node in graph.get_nodes()])
    return X.T
