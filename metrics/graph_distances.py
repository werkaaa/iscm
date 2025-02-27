import numpy as np


# Implementation taken from AVICI
def shd(g, h):
    """
    Computes pairwise Structural Hamming distance, i.e.
    the number of edge insertions, deletions or flips in order to transform one graph to another
        - this means, edge reversals do not double count
        - this means, getting an undirected edge wrong only counts 1

    Args:
        g:  [..., d, d]
        h:  [..., d, d]
    """
    assert g.ndim == h.ndim
    g = g.astype(float)
    h = h.astype(float)
    abs_diff = np.abs(g - h)
    mistakes = abs_diff + np.swapaxes(abs_diff, -2, -1)  # mat + mat.T (transpose of last two dims)

    # ignore double edges
    mistakes_adj = np.where(mistakes > 1, 1, mistakes)

    return np.triu(mistakes_adj).sum((-1, -2))
