# Standardizing Structural Causal Models

This is the code repository for the paper Standardizing Structural Causal Models
([Ormaniec et al.](https://openreview.net/forum?id=aXuWowhIYt&referrer=%5BAuthor%20Console%5D), 2025, ICLR 2025).

Comprehensive code for reproducing the results from the paper can be found in the
[iscm_full](https://github.com/werkaaa/iscm/tree/iscm_full) branch. Here, we introduce the `iscm` library that packages
sampling from iSCMs, SCMs and naively standardized SCMs.

## Library

To install the `iscm` library, simply run:
```
pip install iscm
```

We recommend using the functions in [graph_sampler.py]() and [data_sampler.py](). For an overview of library functionalities,
see [iSCM_Tutorial.ipynb](), which you can directly open in Google Colab:

## Reference
```
@article{ormaniec2025standardizing,
    title={Standardizing Structural Causal Models},
    author={Weronika Ormaniec and Scott Sussex and Lars Lorch and Bernhard Sch{\"o}lkopf and Andreas Krause},
    journal={The Thirteenth International Conference on Learning Representations},
    year={2025}
}
```