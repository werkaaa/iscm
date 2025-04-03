def check_node_number(graph, noise_vars, line):
    assert graph.get_num_nodes() == len(
        noise_vars), f"Graph and noise variances in {line} are incompatible! Number of nodes: {graph.get_num_nodes()}. \
        Length of the variance vector {len(noise_vars)}."
