from typing import List, Optional, Tuple

import numpy as np
import random
import igraph as ig

from data_generation.graph_utils import Graph, topo_sort, has_cycles
from data_generation.mechanisms.linear import sample_weight


def generate_graph_given_schema(schema: List[Tuple[int, int]], weight_range: Tuple[float, float] = None,
                                rng: Optional[np.random.Generator] = None):
    g = Graph()
    if rng is None:
        rng = np.random.default_rng()
    for node_begin, node_end in schema:
        if weight_range is not None:
            w = sample_weight(weight_range[0], weight_range[1], rng=rng)
            g.add_edge(node_begin, node_end, w)
        else:
            g.add_edge(node_begin, node_end)
    return g


def generate_graph_given_schema_with_weights(schema: List[Tuple[int, int]]):
    g = Graph()
    for node_begin, node_end, weight in schema:
        g.add_edge(node_begin, node_end, weight)
    return g


def generate_graph_given_matrix(matrix: np.array):
    g = Graph()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            w = matrix[i, j]
            if w == 0:
                continue
            g.add_edge(i, j, w)

    g.set_num_nodes(matrix.shape[0])
    return g


def generate_uniform_indegree_schema(num_nodes: int, in_degree: int, bound: int,
                                     rng: Optional[np.random.Generator] = None, shuffle: bool = True):
    nodes = list(range(num_nodes))
    if rng is None:
        rng = np.random.default_rng()
    if shuffle:
        rng.shuffle(nodes)
    schema = []
    if bound <= 0:
        bound = num_nodes
    for i in range(len(nodes)):
        if i <= in_degree:
            parents = nodes[:i]
        else:
            parents = rng.choice(nodes[max(0, i - bound):i], size=in_degree, replace=False)
            # parents = nodes[i - in_degree:i]
        for parent in parents:
            schema.append((parent, nodes[i]))

    return schema


def generate_uniform_indegree_graph(num_nodes: int, in_degree: int, bound: int, weight_range: Tuple[float] = None,
                                    rng: Optional[np.random.Generator] = None, shuffle: bool = True):
    if rng is None:
        rng = np.random.default_rng()
    schema = generate_uniform_indegree_schema(num_nodes=num_nodes, in_degree=in_degree, bound=bound, rng=rng,
                                              shuffle=shuffle)
    g = generate_graph_given_schema(schema, weight_range, rng=rng)
    g.set_num_nodes(num_nodes)
    return g


def generate_erdos_renyi_schema(num_nodes: int, p: Optional[float] = None,
                                edges_per_node: Optional[int] = None,
                                rng: Optional[np.random.Generator] = None):
    def logical_xor(a, b):
        return (a and not b) or (not a and b)

    assert logical_xor(p is None,
                       edges_per_node is None), "Pass probability of an edge or the expected number of edges per node."

    if edges_per_node:
        n_edges = edges_per_node * num_nodes
        p = min(n_edges / ((num_nodes * (num_nodes - 1)) / 2), 0.99)

    nodes = list(range(num_nodes))
    if rng is None:
        rng = np.random.default_rng()
    rng.shuffle(nodes)
    schema = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if rng.uniform(0, 1) < p:
                schema.append((nodes[i], nodes[j]))
    return schema


def generate_undirected_scale_free_schema(num_nodes: int, edges_per_node: int, power: int = 1,
                                          rng: Optional[np.random.Generator] = None):
    """Generates undirected scale-free graph and directs it."""
    if rng is None:
        rng = np.random.default_rng()
    random.seed(rng.bit_generator.state["state"]["state"])  # seed pyrandom based on state of numpy rng
    _ = rng.normal()  # advance rng state by 1
    perm = rng.permutation(num_nodes).tolist()
    g = ig.Graph.Barabasi(n=num_nodes, m=edges_per_node, directed=False, power=power).permute_vertices(perm)
    topological_ordering = rng.permutation(num_nodes).tolist()
    undirected_edges = g.to_tuple_list()
    schema = []
    for node_a, node_b in undirected_edges:
        for node in topological_ordering:
            if node == node_a:
                schema.append((node_a, node_b))
                break
            elif node == node_b:
                schema.append((node_b, node_a))
                break
    return schema


def generate_scale_free_schema(num_nodes: int, edges_per_node: int, power: int = 1,
                               rng: Optional[np.random.Generator] = None):
    """Power-law in-degree"""
    if rng is None:
        rng = np.random.default_rng()
    random.seed(rng.bit_generator.state["state"]["state"])  # seed pyrandom based on state of numpy rng
    _ = rng.normal()  # advance rng state by 1
    perm = rng.permutation(num_nodes).tolist()
    g = ig.Graph.Barabasi(n=num_nodes, m=edges_per_node, directed=True, power=power).permute_vertices(perm)
    return g.to_tuple_list()


def generate_scale_free_transposed_schema(num_nodes: int, edges_per_node: int, power: int = 1,
                                          rng: Optional[np.random.Generator] = None):
    """Power-law out-degree"""
    schema = generate_scale_free_schema(num_nodes=num_nodes, edges_per_node=edges_per_node, power=power, rng=rng)
    schema_transposed = [(e, b) for (b, e) in schema]
    return schema_transposed


def generate_scale_free_graph(num_nodes: int, edges_per_node: int, transposed: bool,
                              weight_range: Tuple[float, float] = None,
                              rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random.default_rng()
    if transposed:
        schema = generate_scale_free_transposed_schema(num_nodes=num_nodes, edges_per_node=edges_per_node, rng=rng)
    else:
        schema = generate_scale_free_schema(num_nodes=num_nodes, edges_per_node=edges_per_node, rng=rng)
    g = generate_graph_given_schema(schema, weight_range, rng=rng)
    g.set_num_nodes(num_nodes)
    return g


def generate_undirected_scale_free_graph(num_nodes: int, edges_per_node: int, weight_range: Tuple[float, float] = None,
                                         rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random.default_rng()
    schema = generate_undirected_scale_free_schema(num_nodes=num_nodes, edges_per_node=edges_per_node, rng=rng)
    g = generate_graph_given_schema(schema, weight_range, rng=rng)
    g.set_num_nodes(num_nodes)
    return g


def generate_erdos_renyi_graph(num_nodes: int, p: Optional[float] = None, edges_per_node: Optional[int] = None,
                               weight_range: Tuple[float, float] = None, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random.default_rng()
    schema = generate_erdos_renyi_schema(num_nodes, p=p, edges_per_node=edges_per_node, rng=rng)
    g = generate_graph_given_schema(schema, weight_range, rng=rng)
    g.set_num_nodes(num_nodes)
    return g
