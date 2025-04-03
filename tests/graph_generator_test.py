import unittest

import numpy as np

from iscm.graph_generator import generate_erdos_renyi_schema, generate_uniform_indegree_schema, \
    generate_scale_free_graph, generate_undirected_scale_free_graph
from iscm.graph_utils import has_cycles, Graph


class TestGenerateErdosRenyiSchema(unittest.TestCase):

    def test_result_type(self):
        result = generate_erdos_renyi_schema(10, p=0.5)
        self.assertIsInstance(result, list, "Returned schema should be a list")

    def test_valid_node_range(self):
        result = generate_erdos_renyi_schema(100, p=0.5)
        nodes = set()
        for edge in result:
            nodes.update(edge)
        self.assertEqual(len(nodes), 100, "Nodes should range from 0 to num_nodes - 1")

    def test_probability_range(self):
        p = 0.8
        result = generate_erdos_renyi_schema(50, p=p)
        edge_count = len(result)
        expected_mean_edge_count = p * 50 * 49 / 2  # Expected number of edges in complete graph
        self.assertAlmostEqual(edge_count, expected_mean_edge_count, delta=25,
                               msg=f"Edge count should be close to {expected_mean_edge_count} for p={p}")

    def test_expected_num_edges(self):
        num_nodes = 50
        edges_per_node = 2
        result = generate_erdos_renyi_schema(num_nodes=num_nodes, edges_per_node=edges_per_node)
        edge_count = len(result) / num_nodes
        expected_mean_edge_count = edges_per_node  # Expected number of edges in complete graph
        self.assertAlmostEqual(edge_count, expected_mean_edge_count, delta=0.5,
                               msg=f"Edge count should be close to {expected_mean_edge_count} for edge_per_node={edges_per_node}")

    def test_rng_seed(self):
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=42)
        result1 = generate_erdos_renyi_schema(10, 0.5, rng=rng1)
        result2 = generate_erdos_renyi_schema(10, 0.5, rng=rng2)
        self.assertEqual(result1, result2, "Results should be the same for the same seed")

    def test_acyclic(self):
        for p in [0.2, 0.4, 0.6, 0.8]:
            result = generate_erdos_renyi_schema(50, p)
            g = Graph(schema=result)
            self.assertFalse(has_cycles(g))


class TestScaleFreeGraph(unittest.TestCase):

    def test_num_outgoing_edges(self):
        num_nodes = 10
        edges_per_node = 2
        result = generate_scale_free_graph(num_nodes=num_nodes, edges_per_node=edges_per_node, transposed=False)
        result_matrix = result.get_matrix_representation(True)
        outgoing = np.sum(result_matrix, axis=1)
        self.assertEqual(np.median(outgoing), edges_per_node)
        self.assertEqual(np.max(outgoing), edges_per_node)

    def test_num_incoming_edges(self):
        num_nodes = 10
        edges_per_node = 2
        result = generate_scale_free_graph(num_nodes=num_nodes, edges_per_node=edges_per_node, transposed=True)
        result_matrix = result.get_matrix_representation(True)
        incoming = np.sum(result_matrix, axis=0)
        self.assertEqual(np.median(incoming), edges_per_node)
        self.assertEqual(np.max(incoming), edges_per_node)

    def test_rng_seed(self):
        num_nodes = 50
        edges_per_node = 2
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=42)
        result1 = generate_scale_free_graph(num_nodes=num_nodes, edges_per_node=edges_per_node, transposed=False,
                                            rng=rng1)
        result2 = generate_scale_free_graph(num_nodes=num_nodes, edges_per_node=edges_per_node, transposed=False,
                                            rng=rng2)
        self.assertTrue(np.allclose(result1.get_matrix_representation(boolean=True),
                                    result2.get_matrix_representation(boolean=True)),
                        "Results should be the same for the same seed")

    def test_acyclic(self):
        num_nodes = 50
        for epn in [2, 4, 10, 20]:
            result = generate_scale_free_graph(num_nodes=num_nodes, edges_per_node=epn, transposed=False)
            self.assertFalse(has_cycles(result))

        for epn in [2, 4, 10, 20]:
            result = generate_undirected_scale_free_graph(num_nodes=num_nodes, edges_per_node=epn)
            self.assertFalse(has_cycles(result))


class TestGenerateUniformIndegreeSchema(unittest.TestCase):

    def test_result_type(self):
        result = generate_uniform_indegree_schema(10, 2, 5)
        self.assertIsInstance(result, list, "Returned schema should be a list")

    def test_valid_node_range(self):
        result = generate_uniform_indegree_schema(10, 2, 5)
        nodes = set()
        for edge in result:
            nodes.update(edge)
        self.assertEqual(len(nodes), 10, "All nodes should be included in the schema")

    def test_indegree_constraint(self):
        num_nodes = 10
        in_degree = 2
        result = generate_uniform_indegree_schema(num_nodes, in_degree, 5)
        for i in range(num_nodes):
            count = sum(1 for edge in result if edge[1] == i)
            self.assertLessEqual(count, in_degree, "In-degree constraint should be satisfied for all nodes")

    def test_bound_back_constraint(self):
        bound = 3
        result = generate_uniform_indegree_schema(20, 2, bound, shuffle=False)
        for edge in result:
            self.assertLessEqual(edge[1] - edge[0], bound)

    def test_rng_seed(self):
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=42)
        result1 = generate_uniform_indegree_schema(10, 2, 5, rng=rng1)
        result2 = generate_uniform_indegree_schema(10, 2, 5, rng=rng2)
        self.assertEqual(result1, result2, "Results should be the same for the same seed")


if __name__ == '__main__':
    unittest.main()
