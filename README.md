# Standardizing Structural Causal Models

This is the code repository for the paper Standardizing Structural Causal Models
([Ormaniec et al. 2025](https://openreview.net/forum?id=aXuWowhIYt&referrer=%5BAuthor%20Console%5D), ICLR 2025).

Comprehensive code for reproducing the results from the paper can be found in the
[iscm_full](https://github.com/werkaaa/iscm/tree/iscm_full) branch. Here, we introduce the `iscm` library that packages
sampling from iSCMs, SCMs, and naively standardized SCMs.

## Library

To install the `iscm` library, run:
```
pip install iscm
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
