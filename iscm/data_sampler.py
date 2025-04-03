"""API for sampling from SCMs, iSCMs, and standardized SCMs.

Remark:
We provide functions for systems defined using linear functions and functions sampled from a Gaussian process using
random Fourier features (RFF). There is an important difference to keep in mind while sampling from these two model
families. Sampling linear data requires specifying the exact model, including the linear relationships between variables
through the graph weights. Sampling rff data, on the other hand, implicitly samples functions from a Gaussian process.
"""
from typing import Dict, Tuple, List

import numpy as np

from iscm import closed_form_data_generator
from iscm.graph_utils import Graph
from iscm.data_generator import generate_linear_data_alternate_standardization, generate_linear_data_raw, \
    generate_linear_data_final_standardization, generate_rff_data_alternate_standardization, generate_rff_data_raw, \
    generate_rff_data_final_standardization

GAUSSIAN = 'gaussian'
UNIFORM = 'uniform'

DEFAULT_LENGTH_SCALE_RANGE = (7.0, 10.0)
DEFAULT_OUTPUT_SCALE_RANGE = (10.0, 20.0)


def sample_linear(
        graph: Graph,
        noise_variance: float | Dict[int, float] | List[float] = 1.,
        standardization: None | str = 'internal',
        sample_size: int = 100,
        noise_distribution: str = GAUSSIAN,
        rng: None | np.random.Generator = None,
        closed_form: bool = False,
) -> np.array:
    """Sample data from linear iSCM, SCM or naively standardized SCM according to a graph.

    Args:
        graph: The DAG underlying the system from which we sample. The weights on the graph edges are used to specify
            the linear relationships between variables.
        noise_variance: The variance of the exogenous noise variables. It can be specified as a dictionary or list
            mapping a vertex in the graph to corresponding variance or, if all variances are equal, as a scalar.
        standardization: The standardization method used. Supported options are sampling from an iSCM (`internal`),
            from an SCM (None) or from an SCM with naive standardization ('naive').
        sample_size: The size of the sample to return.
        noise_distribution: The distribution of the exogenous noise variables. Supported options are a zero-centered
            Gaussian distribution (`gaussian`) and a zero-centered uniform distribution (`uniform`).
        rng: An optional random number generator from numpy.
        closed_form: Whether to use a closed-form formulation of the system. If set to `True`: 1) for SCMs this will
            result in sampling through multiplying a noise vector by a full dense weight matrix which is generally
            slower, 2) for iSCMs this will result in computing the implied SCM and sampling from it through multiplying
            a noise vector by a full dense weight matrix of the implied model. Sampling from a closed form system is not
            supported for naively standardized SCMs. Closed form computation is only supported for Gaussian noise.

    Returns: An array with samples of shape `sample_size` times number of graph nodes.
    """

    if closed_form and (standardization not in ['internal', None] or not noise_distribution == GAUSSIAN):
        raise ValueError(
            "Sampling from a system defined in closed-form is possible only for iSCMs and SCMs without standardization."
        )

    if isinstance(noise_variance, float) or isinstance(noise_variance, int):
        noise_vars = {
            node: noise_variance for node in graph.get_nodes()
        }
    else:
        noise_vars = noise_variance

    if standardization == 'internal':
        if closed_form:
            closed_form_data_generator.generate_linear_data_alternate_standardization(
                graph=graph,
                noise_vars=noise_vars,
                num_samples=sample_size,
                rng=rng,
            )

        return generate_linear_data_alternate_standardization(
            graph=graph,
            noise_vars=noise_vars,
            num_samples=sample_size,
            noise_dist=noise_distribution,
            rng=rng,
            return_prestand_variances=False,
        )

    elif standardization == 'naive':
        return generate_linear_data_final_standardization(
            graph=graph,
            noise_vars=noise_vars,
            num_samples=sample_size,
            noise_dist=noise_distribution,
            rng=rng,
            return_prestand_variances=False,
        )

    elif standardization is None:
        if closed_form:
            closed_form_data_generator.generate_linear_data_raw(
                graph=graph,
                noise_vars=noise_vars,
                num_samples=sample_size,
                rng=rng,
            )

        return generate_linear_data_raw(
            graph=graph,
            noise_vars=noise_vars,
            num_samples=sample_size,
            noise_dist=noise_distribution,
            rng=rng,
        )

    raise ValueError(
        f'Standardization {standardization} is not supported. Possible values are: "internal", "naive" and "none".'
    )


def sample_rff(
        graph: Graph,
        noise_variance: float | Dict[int, float] | List[float] = 1.,
        standardization: None | str = 'internal',
        sample_size: int = 100,
        noise_distribution: str = GAUSSIAN,
        length_scale_range: Tuple[float] = DEFAULT_LENGTH_SCALE_RANGE,
        output_scale_range: Tuple[float] = DEFAULT_OUTPUT_SCALE_RANGE,
        rng: None | np.random.Generator = None,
) -> np.array:
    """Sample data from iSCM, SCM or naively standardized SCM with functions sampled from a Gaussian process.

    Args:
        graph: The DAG underlying the system from which we sample. The weights on the graph edges
        noise_variance: The variance of the exogenous noise variables. It can be specified as a dictionary or list
            mapping a vertex in the graph to corresponding variance or, if all variances are equal, as a scalar.
        standardization: The standardization method to be used. Supported options are sampling from an iSCM (`internal`),
            from an SCM (None) or from an SCM with naive standardization ('naive').
        sample_size: The size of the sample to return.
        noise_distribution: The distribution of the exogenous noise variables. Supported options are a zero-centered
            Gaussian distribution (`gaussian`) and a zero-centered uniform distribution (`uniform`).
        length_scale_range: The range from which the length scale of the Gaussian process will be uniformly sampled.
        output_scale_range: The range from which the output scale of the Gaussian process will be uniformly sampled.
        rng: An optional random number generator from numpy.

    Returns: An array with samples of shape `sample_size` times number of graph nodes.
    """

    if isinstance(noise_variance, float) or isinstance(noise_variance, int):
        noise_vars = {
            node: noise_variance for node in graph.get_nodes()
        }
    else:
        noise_vars = noise_variance

    if standardization == 'internal':
        return generate_rff_data_alternate_standardization(
            graph=graph,
            noise_vars=noise_vars,
            num_samples=sample_size,
            noise_dist=noise_distribution,
            length_scale_range=length_scale_range,
            output_scale_range=output_scale_range,
            rng=rng,
        )
    elif standardization == 'naive':
        return generate_rff_data_final_standardization(
            graph=graph,
            noise_vars=noise_vars,
            num_samples=sample_size,
            noise_dist=noise_distribution,
            length_scale_range=length_scale_range,
            output_scale_range=output_scale_range,
            rng=rng,
        )
    elif standardization is None:
        return generate_rff_data_raw(
            graph=graph,
            noise_vars=noise_vars,
            num_samples=sample_size,
            noise_dist=noise_distribution,
            length_scale_range=length_scale_range,
            output_scale_range=output_scale_range,
            rng=rng,
        )

    raise ValueError(
        f'Standardization {standardization} is not supported. Possible values are: "internal", "naive" and "none".'
    )
