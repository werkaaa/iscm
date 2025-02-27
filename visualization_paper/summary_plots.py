import copy
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", message="No GPU automatically detected")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import itertools
import matplotlib
import matplotlib.transforms
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.colors import ListedColormap
import matplotlib.font_manager  # avoids loading error of fontfamily["serif"]

import seaborn as sns
import numpy as np
import pandas as pd
import scipy

from visualization_paper.definitions import *
from visualization_paper import hyperparam_plot_table
from visualization_paper import methods_comp_plot
from visualization_paper.utils import get_filename_from_data_meta
from visualization_paper.utils import map_data_gen_method_names


def benchmark_summary(save_path, method_results_input, dataset_info, only_metrics, ax_width=2.0, ax_height=4.0,
                      method_summaries=False, dump_main=False, dataset_info_skip=[]):
    if not dump_main:
        print("\nbenchmark_summary:", only_metrics if not None else "<all metrics>")
    assert not (dump_main and only_metrics is not None), "should only dump the full df"

    # preprocess results
    metric_results_dict = defaultdict(dict)

    for method, d in method_results_input.items():
        for metr, l in d.items():
            if only_metrics is not None and metr not in only_metrics:
                continue
            metric_results_dict[metr][method] = l
    if only_metrics is not None:
        metric_results = [(metr, metric_results_dict[metr]) for metr in only_metrics if
                          metric_results_dict[metr]]  # to order
    else:
        metric_results = sorted(list(metric_results_dict.items()))

    df = []
    dataset_info_keys = list(dataset_info[method][0].keys())
    for k in dataset_info_skip:
        dataset_info_keys.remove(k)

    for metric, res in metric_results:
        for m, l in res.items():
            for v, dinfo in zip(l, dataset_info[m]):
                dinfo_list = [dinfo[k] for k in dataset_info_keys]
                df.append(tuple(dinfo_list + [metric, m, v]))
    df = pd.DataFrame(df, columns=dataset_info_keys + ["metric", "method", "val"])

    if df.empty:
        warnings.warn(f"\nNo results reported for metrics `{only_metrics}`. "
                      f"Methods: `{list(method_results_input.keys())}` ")
        return

    # dump summary table of metrics
    save_path.mkdir(exist_ok=True, parents=True)
    if dump_main:
        df.to_csv(path_or_buf=(save_path / "main_summary").with_suffix(".csv"), index=False)
        print("Dumped summary df")
        return


def plot_mean_cov(dataset_cov, dataset_info_for_cov, summary_path):
    mpl.rcParams.update(NEURIPS_RCPARAMS_SHORT)

    grouped = defaultdict(list)
    for index, value in enumerate(dataset_info_for_cov):
        name = get_filename_from_data_meta(value)
        grouped[name].append(index)
    dataset_cov = np.array(dataset_cov)
    ticklabels = [i + 1 for i in range(dataset_cov.shape[1])]
    for name, indices in grouped.items():
        sns.set(font_scale=1.05, style="ticks", rc=NEURIPS_RCPARAMS_SHORT)
        # sns.set(font_scale=2.0, style="ticks", rc=NEURIPS_RCPARAMS_SHORT)
        mean_covariances = np.mean(dataset_cov[indices], axis=0)
        sns.heatmap(mean_covariances, annot=True, cmap='Blues', fmt=".2f", yticklabels=ticklabels,
                    xticklabels=ticklabels, vmin=min(0.7, np.min(mean_covariances)))
        plt.savefig(summary_path / f"{name}.pdf", format="pdf", facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        plt.clf()


def define_support_grid(x, bw, cut=3, clip=(0, np.infty), gridsize=200):
    """Create the grid of evaluation points depending for vector x."""
    clip_lo = -np.inf if clip[0] is None else clip[0]
    clip_hi = +np.inf if clip[1] is None else clip[1]
    gridmin = max(x.min() - bw * cut, clip_lo)
    gridmax = min(x.max() + bw * cut, clip_hi)
    return np.linspace(gridmin, gridmax, gridsize)


def plot_induced_noise_variance_distribution(dataset_prestand_vars, dataset_noise_vars, dataset_info_list, summary_path,
                                             ax_width=2.0, ax_height=2.0):
    induced_noise_vars = defaultdict(lambda: defaultdict(list))
    for prestand_vars, noise_vars, info in zip(dataset_prestand_vars, dataset_noise_vars, dataset_info_list):
        wr = ','.join([f'{n}' for n in info["weight_range"]])
        induced_noise_vars[wr][map_data_gen_method_names(info["data_gen_method"])].append(prestand_vars / noise_vars)
        # plt.gca().yaxis.set_major_formatter('{:.1f}'.format)

    figsize = (FIG_SIZE_NEURIPS_TRIPLE[0] * 6 / 5, FIG_SIZE_NEURIPS_TRIPLE[1] * 6 / 5)
    NEURIPS_RCPARAMS["figure.figsize"] = figsize

    mpl.rcParams.update(NEURIPS_RCPARAMS)

    def define_support(x):
        """Create a 1D grid of evaluation points."""
        kde = scipy.stats.gaussian_kde(x)
        bw = np.sqrt(kde.covariance.squeeze())

        cut = 3
        clip = (0, np.infty)
        gridsize = 200

        clip_lo = -np.inf if clip[0] is None else clip[0]
        clip_hi = +np.inf if clip[1] is None else clip[1]
        gridmin = max(x.min() - bw * cut, clip_lo)
        gridmax = min(x.max() + bw * cut, clip_hi)
        return np.linspace(gridmin, gridmax, gridsize)

    def compute_log_kde(sample, support, bw=None):
        log_sample = np.log10(np.sort(sample))
        if bw:
            kde = scipy.stats.gaussian_kde(log_sample, bw_method=bw)
        else:
            kde = scipy.stats.gaussian_kde(log_sample)
        return kde.pdf(support)

    for weight_range, weight_range_dict in induced_noise_vars.items():

        for method in DATA_GEN_METHODS_ORDERING_SHORT:
            all_samples = np.concatenate(weight_range_dict[method])
            support = define_support(np.log10(all_samples))

            densities = []
            for sample in weight_range_dict[method]:
                densities.append(compute_log_kde(sample, support))

            density_mean = np.mean(densities, axis=0)
            density_std = np.std(densities, axis=0, ddof=1)

            support_exp = np.power(10, support)
            plt.plot(support_exp, density_mean, color=DATA_GEN_METHODS_CONFIG[method])
            plt.fill_between(support_exp, density_mean - density_std, density_mean + density_std,
                             color=DATA_GEN_METHODS_CONFIG[method], alpha=0.2)

        plt.xscale('log')
        plt.ylabel('PDF')
        plt.xlabel('$1 / \\theta_i^2$')
        plt.xlim(1.0, 10. ** 4)  # max(all_samples) + 0.1)
        plt.ylim(0)
        plt.xticks([1, 10, 100, 1000])
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        plt.savefig(summary_path / f"{weight_range}_induced_noise_var.pdf", format="pdf", facecolor=None,
                    dpi=DPI, bbox_inches='tight')
        plt.clf()


def plot_experiment(experiment_name, summary_path):
    summary = summary_path / "main_summary.csv"
    if experiment_name in ['iclr-find-hp-notears-noise-var-transfer', 'iclr-find-hp-notears',
                           'iclr-find-hp-notears-big-graphs', 'iclr-find-hp-pc',
                           'iclr-find-hp-pc-big-graphs', 'iclr-find-hp-golem',
                           'iclr-find-hp-golem-big-graphs', 'iclr-find-hp-score-big-graphs',
                           'iclr-find-hp-cam', 'iclr-find-hp-cam-big-graphs']:
        hyperparam_plot_table.visualize_metrics_hp_linear(summary, metric="f1",
                                                    save_path=summary_path, experiment_name=experiment_name)
    elif experiment_name in ['iclr-find-hp-notears-rff', 'iclr-find-hp-notears-rff-big-graphs','iclr-find-hp-pc-rff',
                             'iclr-find-hp-pc-rff-big-graphs', 'iclr-find-hp-golem-rff', 'iclr-find-hp-golem-rff-big-graphs',
                             'iclr-find-hp-cam-rff', 'iclr-find-hp-cam-rff-big-graphs']:
        hyperparam_plot_table.visualize_metrics_hp_rff(summary, metric="f1",
                                                 save_path=summary_path / "hp_comparison.pdf", experiment_name=experiment_name)
    elif experiment_name in ['iclr-compare-methods', 'iclr-compare-methods-big-graphs', 'iclr-compare-methods-no-gaussian',]:

        if 'big' in experiment_name:
            title = ''
            show_metric = False
            xticks = True
        else:
            title = '$\\textsc{Linear}$'
            show_metric = True
            xticks = False

        for m in ["shd", "f1", "precision", "recall"]:
            methods_comp_plot.visualize_metrics_comparison(summary, metric=m, save_path=summary_path,
                                                           title=title, xticks=xticks, show_metric=show_metric)
    elif experiment_name in ['iclr-compare-methods-rff', 'iclr-compare-methods-rff-big-graphs']:

        if 'big' in experiment_name:
            title = ''
            show_metric = False
            xticks = True
        else:
            title = '$\\textsc{Nonlinear}$'
            show_metric = True
            xticks = False

        for m in ["shd", "f1", "precision", "recall"]:
            methods_comp_plot.visualize_metrics_comparison_rff(summary, metric=m, save_path=summary_path,
                                                               title=title, xticks=xticks, show_metric=show_metric)
    elif experiment_name in ['iclr-noise-var-transfer', 'noise-var-transfer']:
        for m in ["shd", "f1", "precision", "recall"]:
            methods_comp_plot.visualize_notears_metrics_comparison(summary, metric=m, save_path=summary_path)
