from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from visualization_paper.definitions import *
from visualization_paper.utils import map_data_gen_method_names
from results_reproducibility.definitions import NOTEARS, PC, GOLEM, GOLEM_NV, GOLEM_EV, SCORE, CAM


def visualize_metrics_hp_linear_(results, hparams, metric, save_path, ax_width=2.0, ax_height=2.0):
    HP1, HP2 = hparams

    weight_ranges = results['weight_range'].unique()
    data_gen_methods = DATA_GEN_METHODS_ORDERING

    best_dict = {"weight distribution": [], "data generation method": [], HP1: [], HP2: [],
                 "median f1": [], "count": []}

    n_rows = len(weight_ranges)
    n_cols = len(data_gen_methods)

    theme = NEURIPS_RCPARAMS
    theme["figure.figsize"] = FIG_SIZE_NEURIPS_SINGLE
    sns.set(style="ticks", rc=NEURIPS_RCPARAMS)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(ax_width * n_cols, ax_height * n_rows))
    if n_rows == 1:
        axs = [axs]
    for i, (weight_range, axrow) in enumerate(zip(weight_ranges, axs)):
        for j, (data_gen_method, ax) in enumerate(zip(data_gen_methods, axrow)):
            results_part = results[
                (results['weight_range'] == weight_range) & (results['data_gen_method'] == data_gen_method)]

            index_to_group_by = [HP1, HP2]
            grouped_results_count = results_part.groupby(index_to_group_by)['val'].count()

            max_count = grouped_results_count.max()
            for key, value in grouped_results_count.iteritems():
                for _ in range(max_count - value):
                    new_row = {index_to_group_by[0]: [key[0]], index_to_group_by[1]: [key[1]], 'val': [0.0]}
                    results_part = pd.concat([results_part, pd.DataFrame(new_row)])

            sns.boxplot(x=HP1, y='val', hue=HP2, data=results_part, ax=ax, showfliers=False)  # ,

            ax.tick_params(axis='both', which='both', length=0)
            if i + 1 != len(weight_ranges):
                plt.setp(ax.get_xticklabels(), visible=False)

            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylim((0.0, 1.0))
            ax.set_xlabel('')
            ax.set_ylabel('')

            grouped_results_part = results_part.groupby(index_to_group_by)['val'].median()
            print(grouped_results_part, 'grouped results part')
            max_index = grouped_results_part.idxmax()
            best_dict["weight distribution"].append(f"Unif_{{\pm}}{weight_range}")
            best_dict["data generation method"].append(data_gen_method)
            best_dict[HP1].append(max_index[0])
            best_dict[HP2].append(max_index[1])
            best_dict["median f1"].append(grouped_results_part.loc[max_index])
            best_dict["count"].append(grouped_results_count.loc[max_index])

    best_df = pd.DataFrame(best_dict)
    print(best_df.to_latex(index=False, float_format="{:.2f}".format))

    # set titles for each row and colum
    for j, col_title in enumerate(data_gen_methods):
        axs[0, j].set_title(col_title)
        axs[-1, j].set_xlabel(HP1)

    for i, row_title in enumerate(weight_ranges):
        axs[i, 0].set_ylabel(row_title, rotation=90, size='large')
        axs[i, -1].set_ylabel(metric, rotation=-90, labelpad=10)
        axs[i, -1].yaxis.set_label_position('right')
        axs[i, -1].tick_params(axis='y', which='both', labelleft=False, labelright=True)

    # tight layout
    fig.tight_layout()

    # generate directory if it doesn't exist
    save_path.parent.mkdir(exist_ok=True, parents=True)

    plt.savefig(save_path, format="pdf", facecolor=None,
                dpi=DPI, bbox_inches='tight')

    plt.close()


def visualize_metrics_hp_linear_golem(results, hparams):
    HP1, HP2, HP3 = hparams

    weight_ranges = results['weight_range'].unique()
    data_gen_methods = DATA_GEN_METHODS_ORDERING

    best_dict = {"weight distribution": [], "data generation method": [], HP1: [], HP2: [], HP3: [],
                 "median f1": [], "count": []}

    for i, weight_range in enumerate(weight_ranges):
        for j, data_gen_method in enumerate(data_gen_methods):
            results_part = results[
                (results['weight_range'] == weight_range) & (results['data_gen_method'] == data_gen_method)]

            index_to_group_by = hparams
            grouped_results_count = results_part.groupby(index_to_group_by)['val'].count()

            max_count = grouped_results_count.max()
            try:
                results_iterator = grouped_results_count.iteritems()
            except:
                results_iterator = grouped_results_count.items()
            for key, value in results_iterator:
                for _ in range(max_count - value):
                    new_row = {index_to_group_by[0]: [key[0]], index_to_group_by[1]: [key[1]], 'val': [0.0]}
                    results_part = pd.concat([results_part, pd.DataFrame(new_row)])

            grouped_results_part = results_part.groupby(index_to_group_by)['val'].median()
            print(grouped_results_part, 'grouped results part')
            max_index = grouped_results_part.idxmax()
            best_dict["weight distribution"].append(f"Unif_{{\pm}}{weight_range}")
            best_dict["data generation method"].append(data_gen_method)
            best_dict[HP1].append(max_index[0])
            best_dict[HP2].append(max_index[1])
            best_dict[HP3].append(max_index[2])
            best_dict["median f1"].append(grouped_results_part.loc[max_index])
            best_dict["count"].append(grouped_results_count.loc[max_index])

    best_df = pd.DataFrame(best_dict)
    print(best_df.to_latex(index=False, float_format="{:.3f}".format))


def visualize_metrics_hp_linear_score_(results, hparams, metric, save_path, ax_width=2.0, ax_height=2.0):
    HP1 = hparams[0]

    weight_ranges = results['weight_range'].unique()
    data_gen_methods = DATA_GEN_METHODS_ORDERING

    best_dict = {"weight distribution": [], "data generation method": [], HP1: [],
                 "median f1": [], "count": []}

    n_rows = len(weight_ranges)
    n_cols = len(data_gen_methods)

    theme = NEURIPS_RCPARAMS
    theme["figure.figsize"] = FIG_SIZE_NEURIPS_SINGLE
    sns.set(style="ticks", rc=NEURIPS_RCPARAMS)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(ax_width * n_cols, ax_height * n_rows))
    if n_rows == 1:
        axs = [axs]
    for i, (weight_range, axrow) in enumerate(zip(weight_ranges, axs)):
        for j, (data_gen_method, ax) in enumerate(zip(data_gen_methods, axrow)):
            results_part = results[
                (results['weight_range'] == weight_range) & (results['data_gen_method'] == data_gen_method)]

            index_to_group_by = [HP1]
            grouped_results_count = results_part.groupby(index_to_group_by)['val'].count()
            print(grouped_results_count, 'grouped results count')

            max_count = grouped_results_count.max()
            for key, value in grouped_results_count.items():
                print(key, 'key')
                for _ in range(max_count - value):
                    new_row = {index_to_group_by[0]: key, 'val': [0.0]}
                    results_part = pd.concat([results_part, pd.DataFrame(new_row)])

            sns.boxplot(x=HP1, y='val', data=results_part, ax=ax, showfliers=False)  # ,

            ax.tick_params(axis='both', which='both', length=0)
            if i + 1 != len(weight_ranges):
                plt.setp(ax.get_xticklabels(), visible=False)

            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylim((0.0, 1.0))
            ax.set_xlabel('')
            ax.set_ylabel('')

            grouped_results_part = results_part.groupby(index_to_group_by)['val'].median()
            print(grouped_results_part, 'grouped results part')
            max_index = grouped_results_part.idxmax()
            best_dict["weight distribution"].append(f"Unif_{{\pm}}{weight_range}")
            best_dict["data generation method"].append(data_gen_method)
            best_dict[HP1].append(max_index)
            best_dict["median f1"].append(grouped_results_part.loc[max_index])
            best_dict["count"].append(grouped_results_count.loc[max_index])

    best_df = pd.DataFrame(best_dict)
    print(best_df.to_latex(index=False, float_format="{:.2f}".format))

    # set titles for each row and colum
    for j, col_title in enumerate(data_gen_methods):
        axs[0][j].set_title(col_title)
        axs[-1][j].set_xlabel(HP1)

    for i, row_title in enumerate(weight_ranges):
        axs[i][0].set_ylabel(row_title, rotation=90, size='large')
        axs[i][-1].set_ylabel(metric, rotation=-90, labelpad=10)
        axs[i][-1].yaxis.set_label_position('right')
        axs[i][-1].tick_params(axis='y', which='both', labelleft=False, labelright=True)

    # tight layout
    fig.tight_layout()

    # generate directory if it doesn't exist
    save_path.parent.mkdir(exist_ok=True, parents=True)

    plt.savefig(save_path, format="pdf", facecolor=None,
                dpi=DPI, bbox_inches='tight')

    plt.close()


def visualize_metrics_hp_linear(results_csv, metric, save_path, experiment_name='notears'):
    print('Visualizing hp search...')

    # read relevant metrics
    results = pd.read_csv(results_csv)
    if 'id' in results.columns and 'data_folder' in results.columns:
        results.drop(columns=['id', 'data_folder'], inplace=True)
    results = results.loc[results['metric'] == metric]
    results['data_gen_method'] = results['data_gen_method'].apply(map_data_gen_method_names)
    columns = list(results.columns)
    to_remove = ['metric', 'weight_range', 'data_gen_method', 'val', 'method']
    for col in to_remove:
        columns.remove(col)
    print(NOTEARS, PC, experiment_name, 'expeirment_name')
    if NOTEARS in experiment_name:
        for params, group in results.groupby(columns):
            key = "_".join([f"{columns[i]}={params[i]}" for i in range(len(columns))]).replace(' ', '')
            save_path_full = save_path / f'notears_hp_search_{key}.pdf'

            # process notears info
            group['lambd'] = group['method'].str.extract(r'lambd=([0-9.]+)').astype(float)
            group['weight threshold'] = group['method'].str.extract(r'wthreshold=([0-9.]+)').astype(float)
            group.drop(columns=['method'], inplace=True)

            visualize_metrics_hp_linear_(group, ['lambd', 'weight threshold'], metric, save_path_full, ax_width=2.0,
                                         ax_height=2.0)
    elif PC in experiment_name:
        for params, group in results.groupby(columns):
            key = "_".join([f"{columns[i]}={params[i]}" for i in range(len(columns))]).replace(' ', '')
            save_path_full = save_path / f'notears_hp_search_{key}.pdf'

            # process PC info
            group['alpha'] = group['method'].str.extract(r'alpha=([0-9.]+)').astype(float)
            group['test'] = group['method'].str.extract(r'test=([a-z_]+)')
            results.drop(columns=['method'], inplace=True)

            visualize_metrics_hp_linear_(group, ['alpha', 'test'], metric, save_path_full,
                                         ax_width=2.0, ax_height=2.0)

    elif GOLEM in experiment_name:
        for params, group in results.groupby(columns):
            key = "_".join([f"{columns[i]}={params[i]}" for i in range(len(columns))]).replace(' ', '')
            save_path_full = save_path / f'notears_hp_search_{key}.pdf'

            # process GOLEM info
            group['lambda1'] = group['method'].str.extract(r'lambd1=([0-9.]+)').astype(float)
            group['lambda2'] = group['method'].str.extract(r'lambd2=([0-9.]+)').astype(float)
            group['weight threshold'] = group['method'].str.extract(r'wthreshold=([0-9.]+)').astype(float)
            group['variance'] = group['method'].str.extract(r'golem_([a-z]+)')
            group.drop(columns=['method'], inplace=True)

            print(GOLEM_EV)
            visualize_metrics_hp_linear_golem(group[group['variance'] == 'ev'],
                                              ['lambda1', 'lambda2', 'weight threshold'])
            print(GOLEM_NV)
            visualize_metrics_hp_linear_golem(group[group['variance'] == 'nv'],
                                              ['lambda1', 'lambda2', 'weight threshold'])
    elif SCORE in experiment_name:
        for params, group in results.groupby(columns):
            key = "_".join([f"{columns[i]}={params[i]}" for i in range(len(columns))]).replace(' ', '')
            save_path_full = save_path / f'notears_hp_search_{key}.pdf'

            # process SCORE info
            group['alpha'] = group['method'].str.extract(r'alpha=([0-9.]+)').astype(float)
            results.drop(columns=['method'], inplace=True)

            visualize_metrics_hp_linear_score_(group, ['alpha'], metric, save_path_full,
                                               ax_width=2.0, ax_height=2.0)
    elif CAM in experiment_name:
        for params, group in results.groupby(columns):
            key = "_".join([f"{columns[i]}={params[i]}" for i in range(len(columns))]).replace(' ', '')
            save_path_full = save_path / f'cam_hp_search_{key}.pdf'

            # process CAM info
            group['alpha'] = group['method'].str.extract(r'alpha=([0-9.]+)').astype(float)
            group['nsplines'] = group['method'].str.extract(r'nsplines=([0-9.]+)').astype(float)
            group['degsplines'] = group['method'].str.extract(r'degsplines=([0-9.]+)').astype(float)
            group.drop(columns=['method'], inplace=True)

            print(CAM)
            visualize_metrics_hp_linear_golem(group, ['alpha', 'nsplines', 'degsplines'])

    else:
        raise NotImplementedError(f'No HP search summary available for experiment: {experiment_name}')


def visualize_metrics_hp_rff(results_csv, metric, save_path, ax_width=2.0, ax_height=2.0, experiment_name=NOTEARS):
    # Read relevant metrics
    results = pd.read_csv(results_csv)
    results = results.loc[results['metric'] == metric]

    if NOTEARS in experiment_name:
        # process NOTEARS info
        results['lambd'] = results['method'].str.extract(r'lambd=([0-9.]+)').astype(float)
        results['weight threshold'] = results['method'].str.extract(r'wthreshold=([0-9.]+)').astype(float)

        visualize_metrics_hp_rff_(results, ['lambd', 'weight threshold'], metric, save_path, ax_width=ax_width,
                                  ax_height=ax_height)
    elif PC in experiment_name:
        # process PC info
        results['alpha'] = results['method'].str.extract(r'alpha=([0-9.]+)').astype(float)
        results['test'] = results['method'].str.extract(r'test=([a-z_]+)')
        results.drop(columns=['method'], inplace=True)

        visualize_metrics_hp_rff_(results, ['alpha', 'test'], metric, save_path, ax_width=ax_width, ax_height=ax_height)

    elif GOLEM in experiment_name:
        # process GOLEM info
        results['lambda1'] = results['method'].str.extract(r'lambd1=([0-9.]+)').astype(float)
        results['lambda2'] = results['method'].str.extract(r'lambd2=([0-9.]+)').astype(float)
        results['weight threshold'] = results['method'].str.extract(r'wthreshold=([0-9.]+)').astype(float)
        results['variance'] = results['method'].str.extract(r'golem_([a-z]+)')
        results.drop(columns=['method'], inplace=True)

        print(GOLEM_EV)
        visualize_metrics_hp_rff_golem(results[results['variance'] == 'ev'], ['lambda1', 'lambda2', 'weight threshold'])
        print(GOLEM_NV)
        visualize_metrics_hp_rff_golem(results[results['variance'] == 'nv'], ['lambda1', 'lambda2', 'weight threshold'])
    elif CAM in experiment_name:
        # process CAM info
        results['alpha'] = results['method'].str.extract(r'alpha=([0-9.]+)').astype(float)
        results['nsplines'] = results['method'].str.extract(r'nsplines=([0-9.]+)').astype(float)
        results['degsplines'] = results['method'].str.extract(r'degsplines=([0-9.]+)').astype(float)
        results.drop(columns=['method'], inplace=True)

        print(CAM)
        visualize_metrics_hp_rff_golem(results, ['alpha', 'nsplines', 'degsplines'])
    else:
        raise NotImplementedError(f'No HP search summary available for experiment: {experiment_name}')


def visualize_metrics_hp_rff_(results, hparams, metric, save_path, ax_width=2.0, ax_height=2.0):
    data_gen_methods = results['data_gen_method'].unique()

    n_cols = len(data_gen_methods)

    HP1, HP2 = hparams

    best_dict = {"data generation method": [], HP1: [], HP2: [],
                 "median f1": [], "count": []}

    # sns.set_theme(style="ticks", rc=NEURIPS_RCPARAMS)
    sns.set(style="ticks", rc=NEURIPS_RCPARAMS)
    fig, axs = plt.subplots(1, n_cols, figsize=(ax_width * n_cols, ax_height))

    for j, (data_gen_method, ax) in enumerate(zip(data_gen_methods, axs)):
        legend = True if j == 0 else False
        results_part = results[results['data_gen_method'] == data_gen_method]

        index_to_group_by = [HP1, HP2]
        grouped_results_count = results_part.groupby(index_to_group_by)['val'].count()
        max_count = grouped_results_count.max()
        for key, value in grouped_results_count.iteritems():
            for _ in range(max_count - value):
                new_row = {index_to_group_by[0]: [key[0]], index_to_group_by[1]: [key[1]], 'val': [0.0]}
                results_part = pd.concat([results_part, pd.DataFrame(new_row)])

        sns.boxplot(x=HP1, y='val', hue=HP2, data=results_part, ax=ax, showfliers=False)
        ax.tick_params(axis='both', which='both', length=0)

        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_ylim(top=1.0)
        ax.set_xlabel('')
        ax.set_ylabel('')
        grouped_results_part = results_part.groupby(index_to_group_by)['val'].median()
        print(grouped_results_part, 'grouped results')
        max_index = grouped_results_part.idxmax()
        best_dict["data generation method"].append(map_data_gen_method_names(data_gen_method))
        best_dict[HP1].append(max_index[0])
        best_dict[HP2].append(max_index[1])
        best_dict["median f1"].append(grouped_results_part.loc[max_index])
        best_dict["count"].append(grouped_results_count.loc[max_index])

    best_df = pd.DataFrame(best_dict)
    print(best_df.to_latex(index=False, float_format="{:.2f}".format))

    # set titles for each row and colum
    for j, col_title in enumerate(data_gen_methods):
        axs[j].set_title(map_data_gen_method_names(col_title))
        axs[j].set_xlabel(HP1)

    axs[-1].set_ylabel(metric, rotation=-90, labelpad=10)
    axs[-1].yaxis.set_label_position('right')
    axs[-1].tick_params(axis='y', which='both', labelleft=False, labelright=True)

    # tight layout
    fig.tight_layout()

    # generate directory if it doesn't exist
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path.with_suffix(".pdf"), format="pdf", facecolor=None,
                dpi=DPI, bbox_inches='tight')

    plt.close()


def visualize_metrics_hp_rff_golem(results, hparams):
    data_gen_methods = results['data_gen_method'].unique()

    HP1, HP2, HP3 = hparams

    best_dict = {"data generation method": [], HP1: [], HP2: [], HP3: [],
                 "median f1": [], "count": []}

    for j, data_gen_method in enumerate(data_gen_methods):
        results_part = results[results['data_gen_method'] == data_gen_method]

        index_to_group_by = hparams
        grouped_results_count = results_part.groupby(index_to_group_by)['val'].count()
        max_count = grouped_results_count.max()

        try:
            results_iterator = grouped_results_count.iteritems()
        except:
            results_iterator = grouped_results_count.items()

        for key, value in results_iterator:
            for _ in range(max_count - value):
                new_row = {index_to_group_by[0]: [key[0]], index_to_group_by[1]: [key[1]], 'val': [0.0]}
                results_part = pd.concat([results_part, pd.DataFrame(new_row)])

        grouped_results_part = results_part.groupby(index_to_group_by)['val'].median()
        print(grouped_results_part, 'grouped results')
        max_index = grouped_results_part.idxmax()
        best_dict["data generation method"].append(map_data_gen_method_names(data_gen_method))
        best_dict[HP1].append(max_index[0])
        best_dict[HP2].append(max_index[1])
        best_dict[HP3].append(max_index[2])
        best_dict["median f1"].append(grouped_results_part.loc[max_index])
        best_dict["count"].append(grouped_results_count.loc[max_index])

    best_df = pd.DataFrame(best_dict)
    print(best_df.to_latex(index=False, float_format="{:.3f}".format))


if __name__ == '__main__':
    visualize_metrics_hp_linear(results_csv="../results/results.csv", metric='f1',
                                save_path=Path("../results/test_plot"))
