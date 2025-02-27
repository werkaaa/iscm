import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import warnings

import networkx as nx
import numpy as np


class Graph:
    def __init__(self, weight_dict: Optional[Dict[Tuple[int, int], float]] = None,
                 weight_matrix: Optional[np.array] = None,
                 schema: Optional[List[Tuple[int, int]]] = None):
        if weight_matrix is not None:
            self.weight_dict = dict()
            for start_node in range(weight_matrix.shape[0]):
                for end_node in range(weight_matrix.shape[1]):
                    if weight_matrix[start_node, end_node] == 0:
                        continue
                    self.weight_dict[(start_node, end_node)] = weight_matrix[start_node, end_node]
        elif weight_dict is not None:
            self.weight_dict = weight_dict
        elif schema is not None:
            self.weight_dict = {edge: None for edge in schema}
        else:
            self.weight_dict = dict()
        self.nodes = set()
        self.children = defaultdict(list)
        self.parents = defaultdict(list)
        for node_start, node_end in self.weight_dict.keys():
            self.nodes.add(node_start)
            self.nodes.add(node_end)
            self.children[node_start].append(node_end)
            self.parents[node_end].append(node_start)

    def get_children(self, node: int):
        return self.children[node]

    def get_parents(self, node: int):
        return self.parents[node]

    def get_roots(self):
        all_children = [child for child, parents in self.parents.items() if len(parents) > 0]
        roots = self.nodes.difference(all_children)
        return roots

    def get_leafs(self):
        leafs = self.nodes.difference(self.children.keys())
        return leafs

    def get_weight(self, node_begin, node_end):
        return self.weight_dict[(node_begin, node_end)]

    def get_matrix_representation(self, boolean: bool = False):
        W = np.zeros((self.get_num_nodes(), self.get_num_nodes()))
        for (node_begin, node_end), w in self.weight_dict.items():
            if boolean:
                W[node_begin, node_end] = True
            else:
                W[node_begin, node_end] = w
        return W

    def get_undirected_matrix_representation(self, boolean: bool = False):
        """This only makes sense for DAGs."""
        W = np.zeros((self.get_num_nodes(), self.get_num_nodes(),))
        for (node_begin, node_end), w in self.weight_dict.items():
            if boolean:
                W[node_begin, node_end] = True
                W[node_end, node_begin] = True
            else:
                W[node_begin, node_end] = w
                W[node_end, node_begin] = w
        return W

    def get_children_gen(self, node: int):
        for child in self.children[node]:
            yield child

    def get_num_nodes(self):
        return len(self.nodes)

    def get_nodes(self):
        return sorted(self.nodes)

    def set_num_nodes(self, num_nodes):
        for i in range(num_nodes):
            self.nodes.add(i)

    def add_edge(self, node_start, node_end, weight=1):
        if (node_start, node_end) in self.weight_dict:
            warnings.warn(f"An edge from {node_start} to {node_end} is being overwritten!")
        else:
            self.children[node_start].append(node_end)
            self.parents[node_end].append(node_start)
        self.weight_dict[(node_start, node_end)] = weight
        self.nodes.add(node_start)
        self.nodes.add(node_end)

    def has_edge(self, node_start, node_end):
        return node_end in self.children[node_start]

    def get_nx_graph(self):
        nx_graph = nx.DiGraph()
        nx_graph.add_weighted_edges_from([(n1, n2, w) for (n1, n2), w in self.weight_dict.items()])
        return nx_graph

    def get_json_dict(self):
        return {f'({node_begin}, {node_end})': w for (node_begin, node_end), w in self.weight_dict.items()}


def topo_sort(graph_or_schema: Union[Graph, Tuple[int, int]]):
    if isinstance(graph_or_schema, Graph):
        G = graph_or_schema
    else:
        G = Graph({edge: 0 for edge in graph_or_schema})

    def topo_sort_subsection(G: Graph, starting_node: int, visited: Dict[int, bool], stack: List[int]):
        working_stack = [(starting_node, G.get_children_gen(starting_node))]
        while len(working_stack) != 0:
            current_node, gen = working_stack.pop()
            visited[current_node] = True
            for child in gen:
                if not visited[child]:
                    working_stack.append((current_node, gen))
                    working_stack.append((child, G.get_children_gen(child)))
                    break
            else:
                # Executes only if for exited with no break
                stack.append(current_node)

    visited = {node: False for node in G.nodes}
    stack = []

    for i in G.nodes:
        if not visited[i]:
            topo_sort_subsection(G, i, visited, stack)
    stack.reverse()
    return stack


def has_cycles(G: Graph):
    visited = {node: False for node in G.nodes}
    being_processed = {node: False for node in G.nodes}

    for i in G.nodes:

        if visited[i]:
            continue

        stack = [i]

        while len(stack) != 0:
            node = stack[-1]

            if not visited[node]:
                visited[node] = True
                being_processed[node] = True
            else:
                stack.pop()
                being_processed[node] = False

            for child in G.get_children(node):
                if not visited[child]:
                    stack.append(child)
                elif being_processed[child]:
                    return True

    return False


def convert_to_binary_matrix(weight_matrix):
    return (weight_matrix != 0).astype(float)


def get_in_degrees(W: np.matrix):
    W_bin = convert_to_binary_matrix(W)
    scores = np.sum(W_bin, axis=0)
    return scores


def get_neighbour_counts(W: np.matrix):
    W_bin = convert_to_binary_matrix(W)
    scores = np.sum(W_bin, axis=0)
    scores += np.sum(W_bin, axis=1)
    return scores


def get_markov_blanket_sizes(W: np.matrix):
    W_bin = convert_to_binary_matrix(W)
    scores = np.zeros(W.shape[0])
    for i in range(len(scores)):
        children = W_bin[i]
        children_ids = children.astype(bool)
        parents = W_bin[:, i]
        parents_of_children = np.sum(W_bin[:, children_ids], axis=1).astype(bool)
        unique_nodes = np.logical_or(np.logical_or(children, parents), parents_of_children)
        unique_nodes[i] = 0
        scores[i] = np.sum(unique_nodes)
    return scores


if __name__ == "__main__":
    # Example code
    g = Graph()
    g.add_edge(5, 2, 0.1)
    g.add_edge(5, 0, 0.1)
    g.add_edge(4, 0, 0.1)
    g.add_edge(4, 1, 0.1)
    g.add_edge(2, 3, 0.1)
    g.add_edge(3, 1, 0.1)
    print(topo_sort(g))
