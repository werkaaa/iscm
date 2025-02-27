from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from visualization_paper.definitions import *
from visualization_paper.utils import map_data_gen_method_names, map_heuristic_data_gen_method_names, \
    map_sortability_name, format_weight_range


def sortability_plot(sortability_results: pd.DataFrame, summary_path: Path):
    figsize = (FIG_SIZE_NEURIPS_DOUBLE[0], FIG_SIZE_NEURIPS_DOUBLE[1] / 2)
    NEURIPS_RCPARAMS["figure.figsize"] = figsize

    sns.set(style="ticks", rc=NEURIPS_RCPARAMS)
    for sortability, results in sortability_results.items():
        print(sortability)
        results_df = pd.DataFrame(results)
        results_df['weight_range'] = results_df['weight_range'].apply(tuple)
        results_df['noise_variance_range'] = results_df['noise_variance_range'].apply(tuple)
        results_df['data_gen_method'] = results_df['data_gen_method'].apply(map_data_gen_method_names)

        keys_to_group = list(results_df.columns)
        keys_to_group.remove('val')
        keys_to_group.remove('data_folder')
        keys_to_group.remove('id')
        keys_to_group.remove('data_gen_method')

        unique_num_nodes = results_df['num_nodes'].unique()
        weight_ranges = results_df['weight_range'].unique()
        unique_edges_per_node = results_df['edges_per_node'].unique()
        if len(unique_num_nodes) > 1:

            keys_to_group.remove('num_nodes')
            keys_to_group.remove('weight_range')

            for params, group in tqdm(results_df.groupby(keys_to_group)):

                fig, axs = plt.subplots(1, len(weight_ranges), sharex=True)

                for i, (w_range, ax) in enumerate(zip(weight_ranges, axs)):
                    results_filtered = group.loc[group['weight_range'] == w_range]
                    sns.lineplot(data=results_filtered, x='num_nodes', y='val', hue='data_gen_method', ax=ax,
                                 hue_order=DATA_GEN_METHODS_ORDERING, palette=DATA_GEN_METHODS_CONFIG, ci='sd')
                    ax.axhline(y=0.5, color='grey', linestyle='--')
                    ax.set_ylim((0.3, 1.0))
                    ax.set_title(format_weight_range(w_range))
                    ax.set_xlabel(None)

                    if i == 0:
                        ax.set_ylabel(map_sortability_name(sortability))

                    if i > 0:
                        ax.yaxis.set_tick_params(left=False, labelleft=False)
                        ax.set_ylabel(None)

                    if i == len(weight_ranges) // 2:
                        ax.set_xlabel('d')
                    ax.get_legend().remove()

                fig.subplots_adjust(wspace=0.1, hspace=0.2)
                config_name = "_".join([f"{keys_to_group[i]}={params[i]}" for i in range(len(keys_to_group))])
                filename = summary_path / f'{sortability}_{config_name}.pdf'
                plt.savefig(filename, format="pdf", facecolor=None,
                            dpi=DPI, bbox_inches='tight')
                plt.close()
        elif len(unique_edges_per_node) > 1:

            keys_to_group.remove('edges_per_node')
            keys_to_group.remove('weight_range')

            for params, group in tqdm(results_df.groupby(keys_to_group)):

                fig, axs = plt.subplots(1, len(weight_ranges), sharex=True)

                for i, (w_range, ax) in enumerate(zip(weight_ranges, axs)):
                    results_filtered = group.loc[group['weight_range'] == w_range]
                    sns.lineplot(data=results_filtered, x='edges_per_node', y='val', hue='data_gen_method', ax=ax,
                                 hue_order=DATA_GEN_METHODS_ORDERING, palette=DATA_GEN_METHODS_CONFIG, ci='sd')
                    ax.axhline(y=0.5, color='grey', linestyle='--')
                    ax.set_ylim((0.3, 1.0))
                    ax.set_title(format_weight_range(w_range))
                    ax.set_xlabel(None)
                    ax.set_xticks([4, 12, 20])

                    if i == 0:
                        ax.set_ylabel(map_sortability_name(sortability))

                    if i > 0:
                        ax.yaxis.set_tick_params(left=False, labelleft=False)
                        ax.set_ylabel(None)

                    if i == len(weight_ranges) // 2:
                        ax.set_xlabel('$k$')
                    ax.get_legend().remove()

                fig.subplots_adjust(wspace=0.1, hspace=0.2)
                config_name = "_".join([f"{keys_to_group[i]}={params[i]}" for i in range(len(keys_to_group))])
                filename = summary_path / f'{sortability}_{config_name}.pdf'
                plt.savefig(filename, format="pdf", facecolor=None,
                            dpi=DPI, bbox_inches='tight')
                plt.close()
        else:
            keys_to_group.remove('weight_range')
            second_groupby = ['weight_range', 'data_gen_method']
            for params, group in results_df.groupby(keys_to_group):
                means_stds = group.groupby(second_groupby)['val'].agg(['mean', 'std'])
                print(means_stds.to_latex(index=True, float_format="{:.2f}".format))


def heuristic_sortability_plot(sortability_results: pd.DataFrame, summary_path: Path):
    figsize = (FIG_SIZE_NEURIPS_DOUBLE[0] / 3, FIG_SIZE_NEURIPS_DOUBLE[1] / 2)
    NEURIPS_RCPARAMS["figure.figsize"] = figsize

    sns.set(style="ticks", rc=NEURIPS_RCPARAMS)
    for sortability, results in sortability_results.items():
        print(sortability)
        results_df = pd.DataFrame(results)
        results_df['weight_range'] = results_df['weight_range'].apply(tuple)
        results_df['noise_variance_range'] = results_df['noise_variance_range'].apply(tuple)
        results_df['data_gen_method'] = results_df['data_gen_method'].apply(map_heuristic_data_gen_method_names)

        results_df = results_df[
            ((results_df['data_gen_method'] == 'Mooij') & (results_df['weight_range'] == (0.5, 1.5))) | (
                        (results_df['data_gen_method'] == 'Squires') & (results_df['weight_range'] == (0.25, 1.0)))]
        keys_to_group = list(results_df.columns)
        keys_to_group.remove('val')
        keys_to_group.remove('data_folder')
        keys_to_group.remove('id')
        keys_to_group.remove('data_gen_method')

        unique_num_nodes = results_df['num_nodes'].unique()
        weight_ranges = results_df['weight_range'].unique()
        print('weight ranges', weight_ranges)
        if len(unique_num_nodes) > 1:

            keys_to_group.remove('num_nodes')
            keys_to_group.remove('weight_range')

            for params, group in tqdm(results_df.groupby(keys_to_group)):
                fig, ax = plt.subplots(1, 1, sharex=True)

                sns.lineplot(data=results_df, x='num_nodes', y='val', hue='data_gen_method', ax=ax,
                             palette=HEURISTIC_DATA_GEN_METHODS_CONFIG, ci='sd')
                ax.axhline(y=0.5, color='grey', linestyle='--')
                ax.set_ylim((0.0, 1.0))
                ax.set_xlabel(None)

                ax.set_ylabel(map_sortability_name(sortability))

                ax.set_xlabel('d')
                ax.get_legend().remove()

            fig.subplots_adjust(wspace=0.1, hspace=0.2)
            config_name = "_".join([f"{keys_to_group[i]}={params[i]}" for i in range(len(keys_to_group))])
            filename = summary_path / f'{sortability}_{config_name}.pdf'
            # plt.legend()
            plt.savefig(filename, format="pdf", facecolor=None,
                        dpi=DPI, bbox_inches='tight')
            plt.close()

        else:
            keys_to_group.remove('weight_range')
            second_groupby = ['weight_range', 'data_gen_method']
            for params, group in results_df.groupby(keys_to_group):
                means_stds = group.groupby(second_groupby)['val'].agg(['mean', 'std'])
                print(means_stds.to_latex(index=True, float_format="{:.2f}".format))
