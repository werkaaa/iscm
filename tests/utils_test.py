from typing import List
import unittest

import numpy as np

from iscm.graph_utils import Graph, has_cycles, topo_sort, get_in_degrees, get_neighbour_counts, get_markov_blanket_sizes


class TestCycleDetection(unittest.TestCase):

    def test_empty(self):
        G = Graph()
        self.assertFalse(has_cycles(G))

    def test_cycles(self):
        # Singe component
        G = Graph(weight_dict={(1, 2): 0, (2, 3): 0, (3, 1): 0, (1, 0): 0})
        self.assertTrue(has_cycles(G))
        # Multiple components
        G = Graph(weight_dict={(1, 2): 0, (2, 3): 0, (3, 1): 0, (1, 0): 0, (4, 5): 0, (4, 6): 0})
        self.assertTrue(has_cycles(G))

    def test_no_cycles(self):
        # Singe component
        G = Graph(weight_dict={(3, 1): 0, (3, 2): 0, (1, 0): 0, (2, 0): 0})
        self.assertFalse(has_cycles(G))
        # Multiple components
        G = Graph(weight_dict={(0, 1): 0, (0, 2): 0, (1, 3): 0, (2, 3): 0, (4, 5): 0, (4, 6): 0})
        self.assertFalse(has_cycles(G))


class TestTopoSort(unittest.TestCase):

    def _check_ordering(self, graph: Graph, ordering: List[int]):
        """Check if there are no back edges in the graph."""
        for i, node_start in enumerate(ordering):
            for node_end in ordering[:i]:
                self.assertFalse(graph.has_edge(node_start, node_end))

    def test_simple(self):
        # path
        g = Graph(schema=[(5, 4), (3, 5), (1, 2), (4, 1)])
        node_sorted = topo_sort(g)
        # node_sorted = [5, 3, 4, 1, 2]
        self._check_ordering(g, node_sorted)

        # tree
        g = Graph(schema=[(6, 5), (5, 4), (5, 3), (4, 2), (2, 1)])
        node_sorted = topo_sort(g)
        self._check_ordering(g, node_sorted)

        # diamond
        g = Graph(schema=[(6, 5), (5, 4), (6, 3), (4, 2), (2, 1), (3, 1)])
        node_sorted = topo_sort(g)
        self._check_ordering(g, node_sorted)

class TestStructuralScores(unittest.TestCase):

    def _get_permutation_matrix(self, size):
        indices = [i for i in range(size)]
        indices = np.random.permutation(indices)
        P = np.eye(size)[indices]
        return P

    def test_get_in_degrees(self):
        # full graph
        W = np.array([[0, 1, 2, 3],
                      [0, 0, 4, 5],
                      [0, 0, 0, 6],
                      [0, 0, 0, 0]])
        P = self._get_permutation_matrix(4)
        self.assertTrue(np.allclose(get_in_degrees(P@W@P.T), P @ np.array([0, 1, 2, 3]), atol=0))

        # some random graph
        W = np.array([[0, 1, 0, 3, 4],
                      [0, 0, 0, 5, 0],
                      [0, 0, 0, 6, 0],
                      [0, 0, 0, 0, 6],
                      [0, 0, 0, 0, 0]])
        P = self._get_permutation_matrix(5)
        self.assertTrue(np.allclose(get_in_degrees(P@W@P.T), P @ np.array([0, 1, 0, 3, 2]), atol=0))

        # chain
        W = np.array([[0, 1, 0, 0, 0],
                      [0, 0, 2, 0, 0],
                      [0, 0, 0, 3, 0],
                      [0, 0, 0, 0, 4],
                      [0, 0, 0, 0, 0]])
        P = self._get_permutation_matrix(5)
        self.assertTrue(np.allclose(get_in_degrees(P@W@P.T), P @ np.array([0, 1, 1, 1, 1]), atol=0))


    def test_get_neighbour_counts(self):
        # full graph
        W = np.array([[0, 1, 2, 3],
                      [0, 0, 4, 5],
                      [0, 0, 0, 6],
                      [0, 0, 0, 0]])
        P = self._get_permutation_matrix(4)
        self.assertTrue(np.allclose(get_neighbour_counts(P @ W @ P.T), P @ np.array([3, 3, 3, 3]), atol=0))

        # some random graph
        W = np.array([[0, 1, 0, 3, 4],
                      [0, 0, 0, 5, 0],
                      [0, 0, 0, 6, 0],
                      [0, 0, 0, 0, 6],
                      [0, 0, 0, 0, 0]])
        P = self._get_permutation_matrix(5)
        self.assertTrue(np.allclose(get_neighbour_counts(P @ W @ P.T), P @ np.array([3, 2, 1, 4, 2]), atol=0))

        # chain
        W = np.array([[0, 1, 0, 0, 0],
                      [0, 0, 2, 0, 0],
                      [0, 0, 0, 3, 0],
                      [0, 0, 0, 0, 4],
                      [0, 0, 0, 0, 0]])
        P = self._get_permutation_matrix(5)
        self.assertTrue(np.allclose(get_neighbour_counts(P @ W @ P.T), P @ np.array([1, 2, 2, 2, 1]), atol=0))

    def test_get_markov_blanket_size(self):
        # full graph
        W = np.array([[0, 1, 2, 3],
                      [0, 0, 4, 5],
                      [0, 0, 0, 6],
                      [0, 0, 0, 0]])
        P = self._get_permutation_matrix(4)
        self.assertTrue(np.allclose(get_markov_blanket_sizes(P @ W @ P.T), P @ np.array([3, 3, 3, 3]), atol=0))

        # some random graph
        W = np.array([[0, 1, 0, 3, 4],
                      [0, 0, 0, 5, 0],
                      [0, 0, 0, 6, 0],
                      [0, 0, 0, 0, 6],
                      [0, 0, 0, 0, 0]])
        P = self._get_permutation_matrix(5)
        self.assertTrue(np.allclose(get_markov_blanket_sizes(P @ W @ P.T), P @ np.array([4, 3, 3, 4, 2]), atol=0))

        # chain
        W = np.array([[0, 1, 0, 0, 0],
                      [0, 0, 2, 0, 0],
                      [0, 0, 0, 3, 0],
                      [0, 0, 0, 0, 4],
                      [0, 0, 0, 0, 0]])
        P = self._get_permutation_matrix(5)
        self.assertTrue(np.allclose(get_markov_blanket_sizes(P @ W @ P.T), P @ np.array([1, 2, 2, 2, 1]), atol=0))


if __name__ == '__main__':
    unittest.main()
