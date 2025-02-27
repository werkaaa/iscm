import json
from types import SimpleNamespace
from typing import Dict, Optional
from pathlib import Path

import numpy as np

from data_generation.data_generator import generate_linear_data_alternate_standardization, \
    generate_linear_data_final_standardization, generate_linear_data_raw, generate_rff_data_alternate_standardization, \
    generate_rff_data_final_standardization, generate_rff_data_raw, generate_linear_data_with_transferred_noise, \
    GAUSSIAN, UNIFORM
from data_generation.heuristic_data_generator import generate_data_squires, generate_data_mooij
from data_generation.graph_generator import generate_graph_given_schema, generate_erdos_renyi_graph, \
    generate_graph_given_schema_with_weights, generate_uniform_indegree_graph, generate_scale_free_graph, \
    generate_undirected_scale_free_graph, generate_graph_given_matrix
from experiment.utils import load_data
from results_reproducibility.definitions import SCHEMAS_SUBDIR, PROJECT_DIR, SCM, iSCM, sSCM, ER, USF, SCHEMA, UNIFORM, \
    SF, SFT, MOOIJ, SQUIRES


def generate(rng: np.random.Generator, num_samples: int, graph_type: str, spec: SimpleNamespace, rff: bool,
             save_vars: bool, induced: bool, heuristic: Optional[str], noise_dist_config=GAUSSIAN) -> Dict:
    """Top function for data generation for experiments."""
    if spec.data_gen_method == SCM:
        # If SCM is passed it is treated as a based system.
        induced = False
    if induced:

        def get_base_system_id(id):
            MOD = 3  # Change accordingly to the number of different methods (SCM, stand. SCM, iSCM).
            if id % MOD == 0:
                return id - 2
            elif id % MOD == 2:
                return id - 1
            else:
                raise ValueError('We assume that the base system goes first!')

        # For induced data we assume that the first data gen method provides graph structure
        id = get_base_system_id(spec.id)
        print(f'Using system with id {id} as a base system for noise transfer.')
        base_data = load_data(Path(spec.data_folder) / f"{id}")

        # We use exactly the same system.
        graph_matrix = base_data['g']
        graph = generate_graph_given_matrix(graph_matrix)

    elif graph_type == ER:
        if rff:
            if 'p' in vars(spec).keys():
                graph = generate_erdos_renyi_graph(num_nodes=spec.num_nodes, p=spec.p, rng=rng)
            else:
                graph = generate_erdos_renyi_graph(num_nodes=spec.num_nodes, edges_per_node=spec.edges_per_node,
                                                   rng=rng)
        else:
            if 'p' in vars(spec).keys():
                graph = generate_erdos_renyi_graph(num_nodes=spec.num_nodes, p=spec.p, weight_range=spec.weight_range,
                                                   rng=rng)
            else:
                graph = generate_erdos_renyi_graph(num_nodes=spec.num_nodes, edges_per_node=spec.edges_per_node,
                                                   weight_range=spec.weight_range, rng=rng)
    elif graph_type == UNIFORM:
        if rff:
            graph = generate_uniform_indegree_graph(num_nodes=spec.num_nodes, in_degree=spec.in_degree,
                                                    bound=spec.bound, rng=rng)
        else:
            graph = generate_uniform_indegree_graph(num_nodes=spec.num_nodes, in_degree=spec.in_degree,
                                                    bound=spec.bound, weight_range=spec.weight_range,
                                                    rng=rng)
    elif graph_type == SF:
        if rff:
            graph = generate_scale_free_graph(num_nodes=spec.num_nodes, edges_per_node=spec.edges_per_node,
                                              transposed=False, rng=rng)
        else:
            graph = generate_scale_free_graph(num_nodes=spec.num_nodes, edges_per_node=spec.edges_per_node,
                                              transposed=False, weight_range=spec.weight_range, rng=rng)
    elif graph_type == SFT:
        if rff:
            graph = generate_scale_free_graph(num_nodes=spec.num_nodes, edges_per_node=spec.edges_per_node,
                                              transposed=True, rng=rng)
        else:
            graph = generate_scale_free_graph(num_nodes=spec.num_nodes, edges_per_node=spec.edges_per_node,
                                              transposed=True, weight_range=spec.weight_range, rng=rng)
    elif graph_type == USF:
        if rff:
            graph = generate_undirected_scale_free_graph(num_nodes=spec.num_nodes, edges_per_node=spec.edges_per_node,
                                                         rng=rng)
        else:
            graph = generate_undirected_scale_free_graph(num_nodes=spec.num_nodes, edges_per_node=spec.edges_per_node,
                                                         weight_range=spec.weight_range, rng=rng)
    elif graph_type == SCHEMA:
        with open(PROJECT_DIR / SCHEMAS_SUBDIR / spec.graph_schema) as f:
            graph_schema = [tuple(x) for x in json.load(f)]

        if "schema_has_weights" in vars(spec).keys() and spec.schema_has_weights:
            graph = generate_graph_given_schema_with_weights(schema=graph_schema)
        else:
            graph = generate_graph_given_schema(schema=graph_schema, weight_range=spec.weight_range, rng=rng)
    else:
        raise NotImplementedError(f"Graph type {graph_type} not known.")

    if induced:
        # We use exactly the same system.
        noise_var = dict(zip(graph.nodes, base_data['noise_vars']))
    else:
        noise_var = dict(zip(graph.nodes, rng.uniform(spec.noise_variance_range[0], spec.noise_variance_range[1],
                                                      graph.get_num_nodes())))

    if heuristic is not None:
        data_gen_method = {
            SQUIRES: lambda g, nv, ns, rng, return_prestand_variances: generate_data_squires(graph=g, num_samples=ns,
                                                                                             rng=rng),
            MOOIJ: lambda g, nv, ns, rng, return_prestand_variances: generate_data_mooij(graph=g, num_samples=ns,
                                                                                         scale_noise=False, rng=rng),
        }
    elif rff:
        print("Generating rff data")

        data_gen_method = {
            SCM: lambda g, nv, ns, rng, return_prestand_variances, noise_dist: generate_rff_data_raw(g, nv, ns,
                                                                                                     length_scale_range=spec.length_scale_range,
                                                                                                     output_scale_range=spec.output_scale_range,
                                                                                                     rng=rng,
                                                                                                     noise_dist=noise_dist),
            sSCM: lambda g, nv, ns, rng, return_prestand_variances, noise_dist: generate_rff_data_final_standardization(
                g, nv,
                ns,
                length_scale_range=spec.length_scale_range,
                output_scale_range=spec.output_scale_range,
                rng=rng,
                noise_dist=noise_dist),
            iSCM: lambda g, nv, ns, rng, return_prestand_variances,
                         noise_dist: generate_rff_data_alternate_standardization(
                g, nv, ns,
                length_scale_range=spec.length_scale_range,
                output_scale_range=spec.output_scale_range,
                rng=rng,
                noise_dist=noise_dist),
        }
    else:
        print("Generating linear data")

        data_gen_method = {
            SCM: lambda g, nv, ns, rng, return_prestand_variances, noise_dist: generate_linear_data_raw(g, nv, ns, rng,
                                                                                                        noise_dist=noise_dist),
            sSCM: generate_linear_data_final_standardization,
            iSCM: generate_linear_data_alternate_standardization
        }

    data = {"sample": data_gen_method[spec.data_gen_method](graph, noise_var, num_samples, rng=rng,
                                                            return_prestand_variances=save_vars or induced,
                                                            noise_dist=noise_dist_config),
            "graph": graph.get_matrix_representation(),
            "noise_vars": np.array([noise_var[node] for node in graph.get_nodes()])}

    if save_vars or induced:
        sample, prestand_variances = data["sample"]

        data["sample"] = sample
        data["prestand_vars"] = prestand_variances

    if induced:
        induced_noise_vars = data["noise_vars"] / data["prestand_vars"]
        data["sample"] = generate_linear_data_with_transferred_noise(graph=graph, noise_vars=noise_var,
                                                                     induced_noise_vars=induced_noise_vars,
                                                                     num_samples=num_samples,
                                                                     rng=rng)

    return data
