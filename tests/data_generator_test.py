import unittest

import numpy as np

from iscm.graph_generator import generate_erdos_renyi_graph, generate_graph_given_schema_with_weights
from iscm.closed_form_data_generator import get_exact_covariance_for_alternate_standardization
from iscm.closed_form_data_generator import \
    generate_linear_data_alternate_standardization as generate_linear_data_alternate_standardization_matrix
from iscm.data_generator import generate_linear_data_alternate_standardization, \
    generate_linear_data_final_standardization, generate_rff_data_alternate_standardization, \
    generate_rff_data_final_standardization, generate_linear_data_raw, generate_linear_data_with_transferred_noise
import iscm


class TestExactDataGeneration(unittest.TestCase):

    def test_data_generation(self):
        print(iscm.data_generator)
        rng = np.random.default_rng(seed=0)
        num_samples = 1000000

        for i in range(3, 6):
            weight_range = (0.5, 2.0)
            noise_var_range = (0.5, 2.0)
            graph = generate_erdos_renyi_graph(num_nodes=i, weight_range=weight_range, p=0.2, rng=rng)
            graph.set_num_nodes(i)

            noise_vars = dict(
                zip(graph.nodes, rng.uniform(noise_var_range[0], noise_var_range[1], graph.get_num_nodes())))

            cov, _, _ = get_exact_covariance_for_alternate_standardization(graph, noise_vars)
            print(np.round(cov, 2))

            data = generate_linear_data_alternate_standardization(graph, noise_vars, num_samples=num_samples, rng=rng)
            cov_data = np.cov(data.T)
            print(np.round(cov_data, 2))

            data2 = generate_linear_data_alternate_standardization_matrix(graph, noise_vars, num_samples=num_samples, rng=rng)
            cov_data2 = np.cov(data2.T)
            print(np.round(cov_data2, 2))

            self.assertTrue(np.allclose(np.round(cov, 2), np.round(cov_data, 2), rtol=0.1))
            self.assertTrue(np.allclose(np.round(cov, 2), np.round(cov_data2, 2), rtol=0.1))

    def test_data_generation_systems_from_paper(self):
        rng = np.random.default_rng(seed=0)
        schema_lists = [
            [[(0, 1, 1), (1, 2, 2)], [(1, 0, 1), (1, 2, 2)], [(1, 0, 1), (2, 1, 2)]],
            [[(1, 0, 1), (2, 1, 2), (2, 3, 3), (2, 4, 4), (5, 4, 5)], [(1, 0, 1), (1, 2, 2), (2, 3, 3), (2, 4, 4), (5, 4, 5)]],
        ]
        noise_var_range = [1, 1]
        for schema_list in schema_lists:
            covariances = []
            for schema in schema_list:
                graph = generate_graph_given_schema_with_weights(schema)
                graph.set_num_nodes(len(schema))

                noise_vars = dict(
                    zip(graph.nodes, rng.uniform(noise_var_range[0], noise_var_range[1], graph.get_num_nodes())))

                cov, _, _ = get_exact_covariance_for_alternate_standardization(graph, noise_vars)
                print(np.round(cov, 2))
                covariances.append(cov)

            for i in range(len(covariances) - 1):
                self.assertTrue(np.allclose(np.round(covariances[i], 2), np.round(covariances[i+1], 2), rtol=0.1))


class TestDataGeneration(unittest.TestCase):

    def test_variance(self):
        rng = np.random.default_rng(seed=0)
        num_samples = 100000

        for i in range(3, 30, 3):
            weight_range = (0.5, 2.0)
            noise_var_range = (0.5, 2.0)
            graph = generate_erdos_renyi_graph(num_nodes=i, weight_range=weight_range, p=0.2, rng=rng)
            graph.set_num_nodes(i)

            noise_vars = dict(
                zip(graph.nodes, rng.uniform(noise_var_range[0], noise_var_range[1], graph.get_num_nodes())))

            data = generate_linear_data_alternate_standardization(graph, noise_vars, num_samples=num_samples, rng=rng)
            self.assertTrue(np.allclose(np.var(data, axis=0), np.ones(i)))

            data = generate_linear_data_final_standardization(graph, noise_vars, num_samples=num_samples, rng=rng)
            self.assertTrue(np.allclose(np.var(data, axis=0), np.ones(i)))

            data = generate_rff_data_alternate_standardization(graph, noise_vars, num_samples=1000000,
                                                               length_scale_range=(7., 10.),
                                                               output_scale_range=(10., 20.), rng=rng)
            self.assertTrue(np.allclose(np.var(data, axis=0), np.ones(i)))

            data = generate_rff_data_final_standardization(graph, noise_vars, num_samples=1000000,
                                                           length_scale_range=(7., 10.), output_scale_range=(10., 20.), rng=rng)
            self.assertTrue(np.allclose(np.var(data, axis=0), np.ones(i)))


class TestNoiseTransfer(unittest.TestCase):

    def test_equal_variance(self):
        num_samples = 10000000
        rng = np.random.default_rng(seed=0)

        for i in range(3, 30, 3):
            weight_range = (0.5, 2.0)
            noise_var_range = (0.8, 1.2)
            graph = generate_erdos_renyi_graph(num_nodes=i, weight_range=weight_range, edges_per_node=2, rng=rng)
            graph.set_num_nodes(i)

            noise_vars = dict(
                zip(graph.nodes, np.ones(graph.get_num_nodes())))
            induced_noise_vars = rng.uniform(noise_var_range[0], noise_var_range[1], graph.get_num_nodes())
            # Ensure that the root nodes have the same variance
            for node in graph.get_roots():
                induced_noise_vars[node] = 1.

            data_original = generate_linear_data_raw(graph, noise_vars, num_samples=num_samples, rng=rng)
            original_vars = np.var(data_original, axis=0, ddof=1)
            data_transferred = generate_linear_data_with_transferred_noise(graph, noise_vars, induced_noise_vars,
                                                                                 num_samples=num_samples, rng=rng)
            transferred_vars = np.var(data_transferred, axis=0, ddof=1)
            print(original_vars)
            print(transferred_vars)
            self.assertTrue(np.allclose(original_vars, transferred_vars, rtol=10**(-2)))


if __name__ == '__main__':
    unittest.main()
