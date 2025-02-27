import copy
import os
import sys
import subprocess
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'meta_data')


def generate_base_command(module, flags=None):
    """ Module is a python file to execute """
    interpreter_script = sys.executable
    base_exp_script = os.path.abspath(module.__file__)
    base_cmd = interpreter_script + ' ' + base_exp_script
    if flags is not None:
        assert isinstance(flags, dict), "Flags must be provided as dict"
        for flag in flags:
            setting = flags[flag]
            base_cmd += f" --{flag}={setting}"
    return base_cmd


def generate_run_commands(command_list=None, array_command=None, array_indices=None, n_cpus=1, n_gpus=1, dry=False,
                          mem=3000, length="short", grouped=False, grouped_length=None,
                          mode='local', gpu_model=None, prompt=True, gpu_mtotal=None,
                          relaunch=False, relaunch_after=None, output_filename="", output_path_prefix="",
                          array_throttle=None):
    # check if single or array
    is_array_job = array_command is not None and array_indices is not None
    assert (command_list is not None) or is_array_job
    if is_array_job:
        assert all([(ind.isdigit() if type(ind) != int else True) for ind in
                    array_indices]), f"array indices must be positive ints but got `{array_indices}`"

    if mode == 'local':
        if prompt and not dry:
            answer = input(f"About to run {len(command_list)} jobs in a loop. Proceed? [yes/no]")
        else:
            answer = 'yes'

        if is_array_job:
            command_list = [array_command.replace("\$SLURM_ARRAY_TASK_ID", str(ind)) for ind in array_indices]

        if answer == 'yes':
            for cmd in command_list:
                # cmd = cmd + '& disown'
                if dry:
                    print(cmd, end="\n\n")
                else:
                    subprocess.call(cmd, shell=True)
    elif mode == 'cluster' and not grouped:  # slurm
        cluster_cmds = []
        slurm_cmd = 'sbatch ' #-A ls_krausea '

        # Wall-clock time  hours:minutes:secs
        if length == "very_long":
            slurm_cmd += f'--time=119:00:00 '
        elif length == "long":
            slurm_cmd += f'--time=23:00:00 '
        elif length == "medium":
            slurm_cmd += f'--time=12:00:00 '
        elif length == "short":
            slurm_cmd += f'--time=3:00:00 '
        else:
            raise NotImplementedError(f"length `{length}` not implemented")

        # CPU memory and CPUs
        slurm_cmd += f'-n {n_cpus} '  # Number of CPUs
        slurm_cmd += f'--mem-per-cpu={mem} '

        # GPU
        if n_gpus > 0:
            if type(gpu_model) == list:
                raise NotImplementedError("pass single gpu specifier correct gpu specifier")

            gpu_model_spec = f'{gpu_model}:' if gpu_model is not None else ""
            slurm_cmd += f'--gpus={gpu_model_spec}{n_gpus} '

            if gpu_mtotal is not None:
                slurm_cmd += f'--gres=gpumem:{gpu_mtotal} '  # makes sure to select GPU with at least this MB memory

        if is_array_job:
            command_list = [array_command]

        for python_cmd in command_list:

            # add job descr
            if "--descr" in python_cmd:
                job_descr = python_cmd.split("--descr ")[1].split(" --")[0].replace("'", "").lstrip(" ").rstrip(" ")
            elif "--run_name" in python_cmd:
                job_descr = python_cmd.split("--run_name ")[1].split(" --")[0].replace("'", "").lstrip(" ").rstrip(" ")
            else:
                # warnings.warn("--run_name/--descr/--name not in python cmd, generating own `-J` slurm job description that could be odd")
                job_descr = python_cmd.split(".py", 1)[1].replace(" ", "_").replace("--", "")

            job_descr = job_descr.replace("\$SLURM_ARRAY_TASK_ID", "%a")
            slurm_cmd_run = slurm_cmd
            slurm_cmd_run += f'-J "{output_filename}{job_descr}" '
            slurm_cmd_run += f'-o "{output_path_prefix}slurm-{output_filename}{job_descr}%a.txt" '

            if is_array_job:
                if type(array_indices) == range:
                    slurm_cmd_run += f'--array {array_indices.start}-{array_indices.stop - 1}'
                    if array_throttle is not None:
                        slurm_cmd_run += f"%{array_throttle}"
                else:
                    slurm_cmd_run += f'--array {",".join([str(ind) for ind in array_indices])}'

            # add relaunch
            if not relaunch:
                cluster_cmds.append(slurm_cmd_run + " --wrap \"" + python_cmd + "\"")
            else:
                relaunch_flags = f" --relaunch True " \
                                 f" --relaunch_str \'" + slurm_cmd_run.replace("\"", "\\\"") + "\' "
                if relaunch_after is None:
                    relaunch_flags += f' --relaunch_after {60 * (119 if length == "very_long" else (23 if length == "long" else 3))} '
                else:
                    relaunch_flags += f' --relaunch_after {relaunch_after} '

                # add datetime to slurm command only after relaunch flags are set
                slurm_cmd_run = slurm_cmd_run.replace(".out", f"_{datetime.datetime.now().strftime('%d-%m-%H:%M')}.out")
                cluster_cmds.append(slurm_cmd_run + " --wrap \"" + python_cmd + relaunch_flags + "\"")

        if prompt and not dry:
            answer = input(f"About to submit {len(cluster_cmds)} compute jobs to the cluster. Proceed? [yes/no]")
        else:
            answer = 'yes'
        if answer == 'yes':
            print()
            for cmd in cluster_cmds:
                if dry:
                    print(cmd, end="\n")
                else:
                    print(cmd, end="\n")
                    os.system(cmd)
    elif mode == 'cluster' and grouped:  # slurm but grouped into a single job
        slurm_cmd_base = 'sbatch ' #-A ls_krausea '

        if grouped_length == "very_long":
            slurm_cmd_base += f'--time=119:00:00 '
        elif grouped_length == "long":
            slurm_cmd_base += f'--time=23:00:00 '
        elif grouped_length == "medium":
            slurm_cmd_base += f'--time=12:00:00 '
        elif grouped_length == "short":
            slurm_cmd_base += f'--time=3:00:00 '
        else:
            slurm_cmd_base += f'--time=23:00:00 '

        # CPU memory and CPUs
        slurm_cmd_base += f'-n {n_cpus} '  # Number of CPUs
        slurm_cmd_base += f'--mem-per-cpu={mem} '

        # It was supposed to be an array job but the runs are too little so we group them into a single run instead
        # and number them as if it was an array job
        command_list_not_grouped = [array_command.replace("\$SLURM_ARRAY_TASK_ID", str(ind)) for ind in array_indices]

        if isinstance(grouped, int):
            print(f"Splitting runs into {grouped} grups")
            per_group = len(command_list_not_grouped) // grouped
            command_list_all = []
            id = 0
            while id * per_group < len(command_list_not_grouped):
                command_list_all.append(
                    command_list_not_grouped[id * per_group: min((id + 1) * per_group, len(command_list_not_grouped))])
                id += 1
        else:
            print(f"Grouping runs into a single group")
            command_list_all = [command_list_not_grouped]

        for id, command_list in enumerate(command_list_all):
            slurm_cmd = copy.copy(slurm_cmd_base)
            python_cmd = command_list[0]
            # add job descr
            if "--descr" in python_cmd:
                job_descr = python_cmd.split("--descr ")[1].split(" --")[0].replace("'", "").lstrip(" ").rstrip(" ")
            elif "--run_name" in python_cmd:
                job_descr = python_cmd.split("--run_name ")[1].split(" --")[0].replace("'", "").lstrip(" ").rstrip(" ")
            else:
                # warnings.warn("--run_name/--descr/--name not in python cmd, generating own `-J` slurm job description that could be odd")
                job_descr = python_cmd.split(".py", 1)[1].replace(" ", "_").replace("--", "")

            job_descr = job_descr.replace("\$SLURM_ARRAY_TASK_ID", "%a")
            slurm_cmd += f'-J "{output_filename}{job_descr}" '
            slurm_cmd += f'-o "{output_path_prefix}slurm-{output_filename}{job_descr}%a.txt" '

            bash_script_path = f"{output_path_prefix}slurm-job-script-{output_filename}{job_descr}-id-{id}.sh"
            os.makedirs(os.path.dirname(bash_script_path), exist_ok=True)
            with open(bash_script_path, 'w') as f:
                f.write("#!/bin/bash\n\n")
                for command in command_list:
                    f.write(command + '\n')

            slurm_cmd += bash_script_path

            if prompt and not dry:
                answer = input(f"About to submit a single compute jobs to the cluster. Proceed? [yes/no]")
            else:
                answer = 'yes'
            if answer == 'yes':
                print()
                if dry:
                    print(slurm_cmd, end="\n")
                else:
                    print(slurm_cmd, end="\n")
                    os.system(slurm_cmd)
