from typing import Dict

import numpy as np

from iscm.graph_utils import Graph


def draw_rff_params(rng, d, length_scale_range, output_scale_range, n_rff=100):
    """Draws random instantiation of rffs"""
    # draw parameters
    ls = rng.uniform(*length_scale_range)
    c = rng.uniform(*output_scale_range)

    # draw rffs
    # [d, n_rff]
    omega_j = rng.normal(loc=0, scale=1.0 / ls, size=(d, n_rff))

    # [n_rff, ]
    b_j = rng.uniform(0, 2 * np.pi, size=(n_rff,))

    # [n_rff, ]
    w_j = rng.normal(loc=0, scale=1.0, size=(n_rff,))

    return dict(
        c=c,
        omega=omega_j,
        b=b_j,
        w=w_j,
        n_rff=n_rff,
    )


def rff_mechanism_matrix(parents, noise, omega, b, w, c, n_rff):
    # feature map phi = cos(omega @ x + b)
    # [..., n_parents, n_rff], [..., n_parents] -> [..., n_rff]
    phi = np.cos(np.einsum('db,...d->...b', omega, parents) + b)

    # f(x) = w @ phi(x)
    # [..., n_rff], [..., n_rff] -> [...]
    f_j = np.sqrt(2.0) * c * np.einsum('b,...b->...', w, phi) / np.sqrt(n_rff)

    # additive noise
    x_j = f_j + noise
    return x_j


def rff_mechanism(node: int, all_generated: Dict, noise: np.array, graph: Graph, omega, b, w, c, n_rff):
    # for compatibility with linear mechanism
    parents = np.array([all_generated[p] for p in graph.get_parents(node)]).T
    return rff_mechanism_matrix(parents, noise, omega, b, w, c, n_rff)
