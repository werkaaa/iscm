"""API for sampling graphs needed for sampling from iSCMs, SCMs, and standardized SCMs."""
from typing import Tuple

import numpy as np

from iscm.graph_utils import Graph
from iscm.graph_generator import generate_erdos_renyi_graph, generate_undirected_scale_free_graph, \
    generate_scale_free_graph


def sample_erdos_renyi(
        num_nodes: int,
        edge_probability: None | float = None,
        edges_per_node: None | int = None,
        weight_range: None | Tuple[float] = None,
        rng: None | np.random.Generator = None
) -> Graph:
    """Sample a random Erdős-Rényi graph with random weights on edges.

    This function first samples a graph and then for each edge uniformly samples a weight from ±`weight_range`. Either
    `edge_probability` or `edges_per_node` should be specified.

    Args:
        num_nodes: How many vertices should the graph have.
        edge_probability: Probability of a single edge being present in the graph.
        edges_per_node: How many edges per vertex there should be in the graph on average.
        weight_range: The weights on graph edges are uniformly sampled from ±`weigh_range`.
        rng: An optional random number generator from numpy.

    Raises:
        ValueError: If both or none of `edges_per_node` and `edge_probability` are specified.

    Returns: A single random graph.
    """

    if (edge_probability is None) == (edges_per_node is None):
        raise ValueError("Exactly ont of `edge_probability` and `edges_per_node` should be specified.")

    return generate_erdos_renyi_graph(
        num_nodes=num_nodes,
        p=edge_probability,
        edges_per_node=edges_per_node,
        weight_range=weight_range,
        rng=rng
    )


def sample_scale_free(
        num_nodes: int,
        edges_per_node: int,
        direct_graph_post_sampling: bool = True,
        transposed: bool = False,
        weight_range: None | Tuple[float] = None,
        rng: None | np.random.Generator = None,
) -> Graph:
    """Sample a random scale-free graph with random weights on edges.

    This function first samples a graph and then for each edge uniformly samples a weight from ±`weight_range`.

    Args:
        num_nodes: How many vertices should the graph have.
        edges_per_node: How many edges per vertex there should be in the graph on average.
        direct_graph_post_sampling: If `True` then first an undirected scale-free graph is sampled that is later ordered
            according to a random topological order. If `False` then an ordered graph is sampled directly through the
            preferential attachment
        transposed: If a directed graph is sampled directly through the preferential attachment, and `transposed` is set
            to `True`, all the edges change the direction.
        weight_range: The weights on graph edges are uniformly sampled from ±`weigh_range`.
        rng: An optional random number generator from numpy.

    Returns: A single random graph.
    """

    if direct_graph_post_sampling:
        return generate_undirected_scale_free_graph(
            num_nodes=num_nodes,
            edges_per_node=edges_per_node,
            weight_range=weight_range,
            rng=rng,
        )

    return generate_scale_free_graph(
        num_nodes=num_nodes,
        edges_per_node=edges_per_node,
        weight_range=weight_range,
        transposed=transposed,
        rng=rng,
    )
