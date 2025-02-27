import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import shutil
from pathlib import Path
import warnings

warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

from launch import generate_run_commands

from results_reproducibility.utils.parse import load_data_config, load_methods_config

from results_reproducibility.definitions import PROJECT_DIR, RESULTS_SUBDIR, EXPERIMENTS_SUBDIR, EXPERIMENT_CONFIG_DATA, \
    EXPERIMENT_CONFIG_METHODS, EXPERIMENT_DATA, EXPERIMENT_PREDS, EXPERIMENT_SUMMARY, YAML_RUN, \
    DEFAULT_RUN_KWARGS, SCRATCH


class ExperimentManager:
    """Tool for clean and reproducible experiment handling via folders"""

    def __init__(self, experiment, seed=0, verbose=True, compute="local", dry=True, num_graphs=None):
        self.experiment = experiment
        self.config_path = PROJECT_DIR / EXPERIMENTS_SUBDIR / self.experiment
        self.store_path_root = PROJECT_DIR
        if SCRATCH is not None:
            self.store_path = SCRATCH / RESULTS_SUBDIR / self.experiment
        else:
            self.store_path = self.store_path_root / RESULTS_SUBDIR / self.experiment
        self.seed = seed
        self.compute = compute
        self.verbose = verbose
        self.dry = dry

        self.lsf_logs_dir = f"{PROJECT_DIR}/logs_lsf/"
        Path(self.lsf_logs_dir).mkdir(exist_ok=True)

        self.data_config_path = self.config_path / EXPERIMENT_CONFIG_DATA
        self.methods_config_path = self.config_path / EXPERIMENT_CONFIG_METHODS

        if self.verbose:
            if self.config_path.exists() \
                    and self.config_path.is_dir() and (self.data_config_path.is_file()
                                                       or (
                                                               self.methods_config_path.is_file() and self.data_config_path.is_file())):
                print("experiment:       ", self.experiment, flush=True)
                print("results directory:", self.store_path, flush=True, end="\n\n")
            else:
                print(f"experiment `{self.experiment}` not specified in `{self.config_path}`."
                      f"check spelling and files")
                exit(1)

        # parse configs
        self.data_config = load_data_config(self.data_config_path, verbose=False, abspath=True)
        self.methods_config = load_methods_config(self.methods_config_path, abspath=True)

        self.num_graphs = self.data_config.num_graphs if num_graphs is None else num_graphs
        self.num_dataconfigs = len(self.data_config.data)

    def _inherit_specification(self, subdir, inherit_from):
        if inherit_from is not None:
            v = str(inherit_from.name).split("_")[1:]
            return subdir + "_" + "_".join(v)
        else:
            return subdir

    def _get_name_without_version(self, p):
        return "_".join(p.name.split("_")[:-1])

    def _list_main_folders(self, subdir, root_path=None, inherit_from=None):
        if root_path is None:
            root_path = self.store_path
        subdir = self._inherit_specification(subdir, inherit_from)
        if root_path.is_dir():
            return sorted([
                p for p in root_path.iterdir()
                if (p.is_dir() and subdir == self._get_name_without_version(p))
            ])
        else:
            return []

    def _init_folder(self, subdir, root_path=None, inherit_from=None, dry=False, add_logs_folder=False):
        if root_path is None:
            root_path = self.store_path
        subdir = self._inherit_specification(subdir, inherit_from)
        existing = self._list_main_folders(subdir, root_path=root_path)

        folder = root_path / (subdir + f"_{len(existing):02d}")
        if not dry:
            folder.mkdir(exist_ok=False, parents=True)
            if add_logs_folder:
                (folder / "logs").mkdir(exist_ok=False, parents=True)
        return folder

    def _copy_file(self, from_path, to_path):
        shutil.copy(from_path, to_path)

    def make_data(self, check=False):
        if check:
            assert self.store_path.exists(), "folder doesn't exist; run `--data` first"
            paths_data = self._list_main_folders(EXPERIMENT_DATA)
            assert len(paths_data) > 0, "data not created yet; run `--data` first"
            final_data = list(filter(lambda p: p.name.rsplit("_", 1)[-1] == "final", paths_data))
            if final_data:
                assert len(final_data) == 1
                return final_data[0]
            else:
                return paths_data[-1]

        # init results folder
        if not self.store_path.exists():
            self.store_path.mkdir(exist_ok=False, parents=True)

        # init data folder
        path_data = self._init_folder(EXPERIMENT_DATA)
        self._copy_file(self.data_config_path, path_data / EXPERIMENT_CONFIG_DATA)
        if self.dry:
            shutil.rmtree(path_data)

        # launch runs that generate data
        experiment_name = kwargs.experiment.replace("/", "--")
        cmd = f"python '{PROJECT_DIR}/experiment/data.py' " \
              f"--j \$SLURM_ARRAY_TASK_ID  " \
              f"--data_config_path '{self.data_config_path}' " \
              f"--path_data '{path_data}' " \
              f"--descr '{experiment_name}-data' "

        generate_run_commands(
            array_command=cmd,
            array_indices=range(1, self.num_graphs * self.num_dataconfigs + 1),
            mode="cluster" if "cluster" in self.compute else self.compute,
            length="short",
            n_cpus=1,
            n_gpus=0,
            prompt=False,
            dry=self.dry,
            output_path_prefix=self.lsf_logs_dir,
            grouped=3,
            grouped_length='medium'
        )
        print(f"\nLaunched {self.num_graphs * self.num_dataconfigs} runs total.")
        return path_data

    def launch_methods(self, check=False, rerun=None):
        # check data has been generated
        path_data = self.make_data(check=True)

        if check:
            paths_results = self._list_main_folders(EXPERIMENT_PREDS, inherit_from=path_data)
            assert len(paths_results) > 0, "results not created yet; run `--methods` first"
            final_results = list(filter(lambda p: p.name.rsplit("_", 1)[-1] == "final", paths_results))
            if final_results:
                assert len(final_results) == 1
                return final_results[0]
            else:
                return paths_results[-1]

        # init results folder
        if rerun is None:
            path_results = self._init_folder(EXPERIMENT_PREDS, inherit_from=path_data)
        else:
            path_results = self.store_path / rerun
        self._copy_file(self.methods_config_path, path_results / EXPERIMENT_CONFIG_METHODS)
        if self.dry:
            shutil.rmtree(path_results)

        # print data sets expected and found
        data_found = sorted([p for p in path_data.iterdir() if p.is_dir()])
        if len(data_found) != self.num_dataconfigs * self.num_graphs:
            warnings.warn(f"Number of data sets does not match data config "
                          f"(got: `{len(data_found)}`, expected `{self.num_dataconfigs * self.num_graphs}`).\n"
                          f"data path: {path_data}\n")
            if len(data_found) < self.num_dataconfigs * self.num_graphs:
                print("Aborting.")
                return
            else:
                print(f"Taking first {self.num_dataconfigs * self.num_graphs} data folders")
                data_found = data_found[:self.num_dataconfigs * self.num_graphs]

        elif self.verbose:
            print(f"\nLaunching experiments for {len(data_found)} data sets.")

        n_launched, n_methods = 0, 0
        path_data_root = data_found[0].parent

        # launch runs that execute methods
        experiment_name = kwargs.experiment.replace("/", "--")
        for k, (method, hparams) in enumerate(self.methods_config.items()):

            n_methods += 1
            seed_indices = sorted([int(p.name) for p in data_found])

            # if possible convert to range for shorter bsub command
            if seed_indices == list(range(seed_indices[0], seed_indices[-1] + 1)):
                seed_indices = range(seed_indices[0], seed_indices[-1] + 1)

            cmd = f"python '{PROJECT_DIR}/experiment/run.py' " \
                  f"--method {method} " \
                  f"--data_id \$SLURM_ARRAY_TASK_ID " \
                  f"--path_results '{path_results}' " \
                  f"--path_data_root '{path_data_root}' " \
                  f"--path_methods_config '{self.methods_config_path}' " \
                  f"--descr '{experiment_name}-{method}-run' "

            run_kwargs = hparams[YAML_RUN] if hparams is not None else DEFAULT_RUN_KWARGS
            cmd_args = dict(
                array_indices=seed_indices,
                mode=self.compute,
                dry=self.dry,
                prompt=False,
                output_path_prefix=f"{path_results}/logs/",
                **run_kwargs,
            )
            # normal method (1 run per dataset)
            n_launched += len(seed_indices)
            generate_run_commands(array_command=cmd, **cmd_args)

        print(f"\nLaunched {n_launched} runs total ({n_methods} methods)")
        return path_results

    def make_summary(self, just_data=False, name_results=None):
        path_data = self.make_data(check=True)
        if not just_data:
            if name_results is None:
                path_results = self.launch_methods(check=True)
            else:
                path_results = self.store_path / name_results

            path_plots = self._init_folder(EXPERIMENT_SUMMARY, inherit_from=path_results)

            if self.dry:
                shutil.rmtree(path_plots)

            # print results expected and found
            methods_config_raw = self.methods_config
            methods_config = {}
            for method in methods_config_raw.keys():
                methods_config[method] = methods_config_raw[method]

            results = sorted([p for p in path_results.iterdir()])
            results_found = {}
            n_results_found = 0
            for method, _ in methods_config.items():
                n_expected = self.num_graphs * self.num_dataconfigs
                method_results = list(filter(lambda p: p.name.rsplit("_", 1)[0] == method, results))
                results_found[method] = method_results
                n_results_found += len(method_results)
                print(f"{method + ':':30s}{len(method_results):4d}/{n_expected}\t"
                      f"{'(!)' if len(method_results) != n_expected else ''}")

            if not n_results_found:
                return

        else:
            path_results = ''
            path_plots = self._init_folder(EXPERIMENT_SUMMARY, inherit_from=path_data)

        # create summary
        experiment_name = kwargs.experiment.replace("/", "--")
        cmd = f"python '{PROJECT_DIR}/experiment/summary.py' " \
              f"--methods_config_path {self.methods_config_path} " \
              f"--path_data {path_data} " \
              f"--path_plots '{path_plots}' " \
              f"--path_results '{path_results}' " \
              f"--experiment_name '{self.experiment}' " \
              f"--descr '{experiment_name}-{path_plots.parts[-1]}' "

        generate_run_commands(
            command_list=[cmd],
            mode="cluster" if "cluster" in self.compute else self.compute,
            length="short",
            n_gpus=0,
            n_cpus=4,
            mem=3000,
            prompt=False,
            dry=self.dry,
            output_path_prefix=self.lsf_logs_dir,
        )
        return path_plots


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, nargs="?", default="test", help="experiment config folder")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--compute", type=str, default="local")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--methods", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--just_data", action="store_true", default=False)
    parser.add_argument("--name_results", type=str, default=None)
    parser.add_argument("--num_graphs", type=int, help="overwrites default specified in config")
    parser.add_argument("--rerun", type=str, default=None, help='name of the folder with partial results')

    kwargs = parser.parse_args()

    kwargs_sum = sum([
        kwargs.data,
        kwargs.methods,
        kwargs.summary,
    ])
    assert kwargs_sum == 1, f"pass 1 option, got `{kwargs_sum}`"

    exp = ExperimentManager(experiment=kwargs.experiment, compute=kwargs.compute, num_graphs=kwargs.num_graphs,
                            dry=not kwargs.submit)

    if kwargs.data:
        _ = exp.make_data()
    elif kwargs.methods:
        _ = exp.launch_methods(rerun=kwargs.rerun)
    elif kwargs.summary:
        _ = exp.make_summary(just_data=kwargs.just_data, name_results=kwargs.name_results)
    else:
        raise ValueError()
