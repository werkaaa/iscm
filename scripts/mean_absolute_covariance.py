import os
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from data_generation.graph_generator import generate_graph_given_schema
from data_generation.data_generator import generate_linear_data_alternate_standardization, \
    generate_linear_data_final_standardization
from results_reproducibility.definitions import RNG_ENTROPY_EVAL, RESULTS_SUBDIR
from visualization_paper.definitions import NEURIPS_RCPARAMS_SHORT, DPI


def generate_mean_abs_covariance(graph_schema, weight_range, noise_var_range, data_gen_func, rng, ax, title="",
                                 num_samples=100000,
                                 num_graphs=100000):
    mpl.rcParams.update(NEURIPS_RCPARAMS_SHORT)
    sns.set(font_scale=0.75, style="ticks", rc=NEURIPS_RCPARAMS_SHORT)
    covariances = dict()
    for i in range(num_graphs):
        g = generate_graph_given_schema(schema=graph_schema, weight_range=weight_range, rng=rng)
        n = dict(zip(g.nodes, rng.uniform(noise_var_range[0], noise_var_range[1], g.get_num_nodes())))
        columns = [i + 1 for i in range(g.get_num_nodes())]
        data = pd.DataFrame(data_gen_func(graph=g, noise_vars=n, num_samples=num_samples, rng=rng), columns=columns)

        covariances[i] = data.cov().abs()
    data = pd.concat(covariances.values(), keys=covariances.keys()).groupby(level=1).mean()
    sns.heatmap(data, annot=True, cmap='Blues', fmt=".2f", ax=ax, cbar=False)
    ax.set_title(title)


if __name__ == '__main__':
    rng = np.random.default_rng(np.random.SeedSequence(entropy=(RNG_ENTROPY_EVAL)))

    # 10 node chain
    graph_schema = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
    weight_range = (0.5, 2.0)
    noise_range = (1.0, 1.0)

    plot_path = Path(RESULTS_SUBDIR) / 'mean_absolute_covariance/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    fig, axs = plt.subplots(ncols=3, figsize=(8.25, 3.5), gridspec_kw=dict(width_ratios=[4, 4, 0.25]))

    generate_mean_abs_covariance(graph_schema, weight_range, noise_range, generate_linear_data_final_standardization,
                                 rng=rng,
                                 title="Standardized SCM", ax=axs[0])
    generate_mean_abs_covariance(graph_schema, weight_range, noise_range,
                                 generate_linear_data_alternate_standardization, rng=rng,
                                 title="iSCM", ax=axs[1])
    cb = fig.colorbar(axs[1].collections[0], cax=axs[2])
    cb.outline.set_edgecolor('white')

    filename = f"10_node_chain.pdf"
    print('Saving the plot to ', plot_path / filename)
    plt.savefig(plot_path / filename, format="pdf", facecolor=None,
                dpi=DPI, bbox_inches='tight')
    plt.clf()
