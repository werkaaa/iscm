# Standardizing Structural Causal Models

![PyPi](https://img.shields.io/pypi/v/iscm?logo=PyPI) [![DOI](https://zenodo.org/badge/939681347.svg)](https://doi.org/10.5281/zenodo.15697868)


This is the code repository for the paper *Standardizing Structural Causal Models*
([Ormaniec et al., 2025](https://openreview.net/forum?id=aXuWowhIYt&referrer=%5BAuthor%20Console%5D), ICLR 2025).

Comprehensive code for reproducing the results from the paper can be found in the
[iscm_full](https://github.com/werkaaa/iscm/tree/iscm_full) branch. Here, we introduce the `iscm` library that packages
sampling from iSCMs, SCMs, and naively standardized SCMs.

## Library

To install the `iscm` library, run:
```
pip install iscm
```

The code snippet below shows how you can sample from an iSCM.
```python
import numpy as np

from iscm import data_sampler, graph_sampler

rng = np.random.default_rng(seed=0)

# Generate a graph
graph = graph_sampler.generate_erdos_renyi_graph(
            num_nodes=20,
            edges_per_node=2,
            weight_range=(0.5, 2.0), # The weights will be sampled randomly from ± weight range
            rng=rng,
        )

# Sample data
iscm_sample = data_sampler.sample_linear(
                  graph=graph,
                  sample_size=100,
                  standardization='internal',
                  rng=rng,
              )
```

We recommend using the functions in [graph_sampler.py](https://github.com/werkaaa/iscm/blob/main/iscm/graph_sampler.py) and [data_sampler.py](https://github.com/werkaaa/iscm/blob/main/iscm/data_sampler.py) to sample graphs and data.
For an overview of library functionalities, see [iSCM_Tutorial.ipynb](https://github.com/werkaaa/iscm/blob/main/iSCM_Tutorial.ipynb), which you can directly open in Google Colab:

<a target="_blank" href="https://colab.research.google.com/github/werkaaa/iscm/blob/main/iSCM_Tutorial.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Reference
```
@article{ormaniec2025standardizing,
    title={Standardizing Structural Causal Models},
    author={Weronika Ormaniec and Scott Sussex and Lars Lorch and Bernhard Sch{\"o}lkopf and Andreas Krause},
    journal={The Thirteenth International Conference on Learning Representations},
    year={2025}
}
```
