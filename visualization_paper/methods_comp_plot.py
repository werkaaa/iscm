from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.colors import ListedColormap
import matplotlib.font_manager  # avoids loading error of fontfamily["serif"]

import seaborn as sns
import pandas as pd

from visualization_paper.definitions import *
from visualization_paper.utils import map_data_gen_method_names, map_method_names, format_metric, format_weight_range
from results_reproducibility.definitions import SCRATCH, AVICI, NOTEARS, VAR_SORT_REGRESS, R2_SORT_REGRESS, RANDOM_SORT_REGRESS, PC, GES, GOLEM_NV, GOLEM_EV, LINGAM, CAM


def visualize_notears_metrics_comparison(results_csv, metric, save_path):
    figsize = (FIG_SIZE_NEURIPS_TRIPLE[0] * 6 / 5, FIG_SIZE_NEURIPS_TRIPLE[1] * 6 / 5)
    NEURIPS_RCPARAMS["figure.figsize"] = figsize

    results = pd.read_csv(results_csv)
    if 'id' in results.columns and 'data_folder' in results.columns:
        results.drop(columns=['id', 'data_folder'], inplace=True)
    results = results.loc[results['metric'] == metric]

    results["method"] = results["method"].apply(map_method_names)

    weight_ranges = results['weight_range'].unique()
    data_gen_methods = results['data_gen_method'].unique()

    def map_data_gen_method_names(data_gen_method):
        if data_gen_method in ['no', 'raw']:
            return 'Original'
        elif data_gen_method in ['alternate']:
            return 'Noise from iSCM'
        elif data_gen_method in ['final']:
            return 'Noise from stand. SCM'
        else:
            raise NotImplementedError(f'Data gen method {data_gen_method} not known')

    DATA_GEN_METHODS_ORDERING = ['Original', 'Noise from stand. SCM', 'Noise from iSCM']

    DATA_GEN_METHODS_CONFIG = {
        "Original": COLORS[0],
        "Noise from stand. SCM": COLORS[1],
        "Noise from iSCM": COLORS[2]
    }

    for dgm in data_gen_methods:
        results = results.replace(dgm, map_data_gen_method_names(dgm))

    for i, weight_range in enumerate(weight_ranges):
        results_part = results[results['weight_range'] == weight_range]

        sns.set(style="ticks", rc=NEURIPS_RCPARAMS)

        sns.boxplot(hue='data_gen_method', y='val', data=results_part,
                    hue_order=DATA_GEN_METHODS_ORDERING, palette=DATA_GEN_METHODS_CONFIG, fliersize=OUTLIER_SIZE,
                    flierprops={"marker": "d"})

        if metric in ['f1', 'precision', 'recall']:
            plt.ylim(0.0, 1.0)

        plt.xlabel('')
        plt.legend().remove()
        plt.ylabel(format_metric(metric))
        plt.tight_layout()
        # plt.title('$\\textsc{Notears}$ (Trans. Noise)')
        plt.title('$\\textsc{Notears}$\n(SCMs with impl. noise scales)')

        save_path.parent.mkdir(exist_ok=True, parents=True)
        weight_str = weight_range.replace(' ', '').replace('[', '').replace(']', '').replace(',', '_')
        plt.savefig(save_path / f"method_comparison_{metric}_w={weight_str}.pdf",
                    format="pdf", facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        plt.close()


def visualize_metrics_comparison(results_csv, metric, save_path, title='', ylabel='', legend=False, xticks=True,
                                 show_metric=True):
    NUM_METHODS_SCALING = 9/5
    if not xticks:
        figure_size = (FIG_SIZE_NEURIPS_TRIPLE[0] * NUM_METHODS_SCALING, FIG_SIZE_NEURIPS_TRIPLE[1] * VERTICAL_SCALING_COMP_TOP)
    else:
        figure_size = (FIG_SIZE_NEURIPS_TRIPLE[0] * NUM_METHODS_SCALING, FIG_SIZE_NEURIPS_TRIPLE[1] * VERTICAL_SCALING_COMP_BOTTOM)
    NEURIPS_RCPARAMS["figure.figsize"] = figure_size

    # read relevant metrics
    results = pd.read_csv(results_csv)
    if 'id' in results.columns and 'data_folder' in results.columns:
        results.drop(columns=['id', 'data_folder'], inplace=True)
    results = results.loc[results['metric'] == metric]

    results["method"] = results["method"].apply(map_method_names)

    weight_ranges = results['weight_range'].unique()
    data_gen_methods = results['data_gen_method'].unique()
    for dgm in data_gen_methods:
        results = results.replace(dgm, map_data_gen_method_names(dgm))

    for i, weight_range in enumerate(weight_ranges):
        results_part = results[results['weight_range'] == weight_range]
        print(results_part, 'results_part')
        sns.set(style="ticks", rc=NEURIPS_RCPARAMS)
        ax = sns.boxplot(x='method', y='val', hue='data_gen_method', data=results_part,
                         hue_order=DATA_GEN_METHODS_ORDERING, order=[map_method_names(m) for m in
                                                                     [NOTEARS, GOLEM_EV, GOLEM_NV, AVICI, PC, GES, CAM, VAR_SORT_REGRESS, R2_SORT_REGRESS,
                                                                      RANDOM_SORT_REGRESS]],
                         fliersize=OUTLIER_SIZE,
                         flierprops={"marker": "d"}, linewidth=1.)

        tl = plt.gca().get_xticks()
        plt.axvline((tl[6] + tl[7]) / 2, color='grey', lw=0.5, ls='--')

        if metric in ['f1', 'precision', 'recall']:
            plt.ylim(0.0, 1.0)

        if legend:
            if metric in ['f1', 'precision', 'recall']:
                plt.legend(title='', loc='upper right')
            else:
                plt.legend(title='', loc='upper left')
        else:
            plt.legend().remove()

        plt.xlabel('')
        if xticks:
            plt.xticks(rotation=90)
        else:
            plt.gca().set_xticklabels([])

        if show_metric:
            ymax = ax.get_ylim()[1]
            plt.text(-1.2, 1.1 * ymax, format_metric(metric), ha='left')

        plt.ylabel(ylabel)
        if weight_range in WEIGHT_RANGE_TITLES and len(title):
            plt.title(format_weight_range(weight_range))
        elif len(title):
            plt.title(title)
        plt.tight_layout()

        # # TODO: Remove when running on scratch
        # save_path = save_path.relative_to(SCRATCH)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        weight_str = weight_range.replace(' ', '').replace('[', '').replace(']', '').replace(',', '_')
        plt.savefig(save_path / f"method_comparison_{metric}_w={weight_str}.pdf",
                    format="pdf", facecolor=None,
                    dpi=DPI, bbox_inches='tight')

        plt.close()


def visualize_metrics_comparison_rff(results_csv, metric, save_path, title='', ylabel='', legend=False, xticks=True,
                                     show_metric=True):
    #TODO: Clean up after rebuttal
    # if not xticks:
    #     figure_size = (FIG_SIZE_NEURIPS_TRIPLE[0], FIG_SIZE_NEURIPS_TRIPLE[1] * VERTICAL_SCALING_COMP_TOP)
    # else:
    #     figure_size = (FIG_SIZE_NEURIPS_TRIPLE[0], FIG_SIZE_NEURIPS_TRIPLE[1] * VERTICAL_SCALING_COMP_BOTTOM)
    NUM_METHODS_SCALING = 9/5
    if not xticks:
        figure_size = (FIG_SIZE_NEURIPS_TRIPLE[0] * NUM_METHODS_SCALING, FIG_SIZE_NEURIPS_TRIPLE[1] * VERTICAL_SCALING_COMP_TOP)
    else:
        figure_size = (FIG_SIZE_NEURIPS_TRIPLE[0] * NUM_METHODS_SCALING, FIG_SIZE_NEURIPS_TRIPLE[1] * VERTICAL_SCALING_COMP_BOTTOM)
    NEURIPS_RCPARAMS["figure.figsize"] = figure_size

    # read relevant metrics
    results = pd.read_csv(results_csv)
    results = results.loc[results['metric'] == metric]

    results["method"] = results["method"].apply(map_method_names)

    data_gen_methods = results['data_gen_method'].unique()
    for dgm in data_gen_methods:
        results = results.replace(dgm, map_data_gen_method_names(dgm, short=True))

    sns.set(style="ticks", rc=NEURIPS_RCPARAMS)

    print(results, 'results')

    ax = sns.boxplot(x='method', y='val', hue='data_gen_method', data=results,
                     hue_order=DATA_GEN_METHODS_ORDERING_SHORT_STAND, order=[map_method_names(m) for m in
                                                                             # [GOLEM_EV, GOLEM_NV]],
                                                                             [NOTEARS, GOLEM_EV, GOLEM_NV, AVICI, PC, GES, CAM, VAR_SORT_REGRESS,
                                                                              R2_SORT_REGRESS,
                                                                              RANDOM_SORT_REGRESS]],
                     fliersize=OUTLIER_SIZE, flierprops={"marker": "d"}, linewidth=1.)  # , showfliers=False)
    tl = plt.gca().get_xticks()
    plt.axvline((tl[6] + tl[7]) / 2, color='grey', lw=0.5, ls='--')

    if metric in ['f1', 'precision', 'recall']:
        plt.ylim(0.0, 1.0)

    if legend:
        if metric in ['f1', 'precision', 'recall']:
            plt.legend(title='', loc='upper right')
        else:
            plt.legend(title='', loc='upper left')
    else:
        plt.legend().remove()

    plt.xlabel('')
    if xticks:
        plt.xticks(rotation=90)
    else:
        plt.gca().set_xticklabels([])

    if show_metric:
        ymax = ax.get_ylim()[1]
        plt.text(-1.2, 1.1 * ymax, format_metric(metric), ha='left')

    plt.ylabel(ylabel)
    if len(title):
        plt.title(title)
    plt.tight_layout()

    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path / f"method_comparison_{metric}.pdf",
                format="pdf", facecolor=None,
                dpi=DPI, bbox_inches='tight')

    plt.close()
