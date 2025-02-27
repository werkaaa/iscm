# Standardizing Structural Causal Models

## Installing requirements

Before running the experiments, create virtual environment that uses Python 3.10 and install requirements:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To run AVICI we also need to download model checkpoints. You can do that by running:

```
source graph_discovery_algo/get_avici_checkpoints.sh 
```

To run PC and GES one needs to have R installed. We used R 4.3.2. We provide an R script that installs all needed R
packages. It can be run with:

```
Rscript rsetup.R
```

Finally, one also needs to install the attached copy of the CDT library:

```
pip install -e ./graph_discovery_algo/cdt-source/
```

## Running experiments

Running all the commands from this section reproduces all figures and tables from the paper.
The results were originally generated on a cluster with slurm. If you want to run them locally just
remove `--compute=cluster` from the commands. All commands should be ran from the main project directory. Every
experiment will create its subdirectory in the automatically created `results` folder. The plots will be saved into
the `summary` subfolder of each experiment.

### Sortabilities

```
python scripts/eval.py iclr-sortabilities-ER-2 --data --submit
python scripts/eval.py iclr-sortabilities-ER-2 --method --submit --compute=cluster
python scripts/eval.py iclr-sortabilities-ER-2 --summary --submit
```

```
python scripts/eval.py iclr-sortabilities-ER-4 --data --submit
python scripts/eval.py iclr-sortabilities-ER-4 --method --submit --compute=cluster
python scripts/eval.py iclr-sortabilities-ER-4 --summary --submit
```

```
python scripts/eval.py iclr-sortabilities-USF-2 --data --submit
python scripts/eval.py iclr-sortabilities-USF-2 --method --submit --compute=cluster
python scripts/eval.py iclr-sortabilities-USF-2 --summary --submit
```

```
python scripts/eval.py iclr-sortabilities-USF-4 --data --submit
python scripts/eval.py iclr-sortabilities-USF-4 --method --submit --compute=cluster
python scripts/eval.py iclr-sortabilities-USF-4 --summary --submit
```

```
python scripts/eval.py iclr-sortabilities-trees --data --submit
python scripts/eval.py iclr-sortabilities-trees --method --submit --compute=cluster
python scripts/eval.py iclr-sortabilities-trees --summary --submit
```

### Strcture learning

#### Finding hyperparameters for NOTEARS

```
python scripts/eval.py iclr-find-hp-notears --data --submit
python scripts/eval.py iclr-find-hp-notears --method --submit --compute=cluster
python scripts/eval.py iclr-find-hp-notears --summary --submit
```

```
python scripts/eval.py iclr-find-hp-notears-big-graphs --data --submit
python scripts/eval.py iclr-find-hp-notears-big-graphs --method --submit --compute=cluster
python scripts/eval.py iclr-find-hp-notears-big-graphs --summary --submit
```

```
python scripts/eval.py iclr-find-hp-notears-rff --data --submit
python scripts/eval.py iclr-find-hp-notears-rff --method --submit --compute=cluster
python scripts/eval.py iclr-find-hp-notears-rff --summary --submit
```

```
python scripts/eval.py iclr-find-hp-notears-rff-big-graphs --data --submit
python scripts/eval.py iclr-find-hp-notears-rff-big-graphs --method --submit --compute=cluster
python scripts/eval.py iclr-find-hp-notears-rff-big-graphs --summary --submit
```

#### Finding hyperparameters for PC algorithm

```
python scripts/eval.py iclr-find-hp-pc --data --submit
python scripts/eval.py iclr-find-hp-pc --method --submit --compute=cluster
python scripts/eval.py iclr-find-hp-pc-big-graphs --data --submit
python scripts/eval.py iclr-find-hp-pc-big-graphs --method --submit --compute=cluster
python scripts/eval.py iclr-find-hp-pc-rff --data --submit
python scripts/eval.py iclr-find-hp-pc-rff --method --submit --compute=cluster
python scripts/eval.py iclr-find-hp-pc-rff-big-graphs --data --submit
python scripts/eval.py iclr-find-hp-pc-rff-big-graphs --method --submit --compute=cluster
```

#### Finding hyperparameters for CAM algorithm

```
python scripts/eval.py iclr-find-hp-cam --data --submit
python scripts/eval.py iclr-find-hp-cam --method --submit --compute=cluster
python scripts/eval.py iclr-find-hp-cam-big-graphs --data --submit
python scripts/eval.py iclr-find-hp-cam-big-graphs --method --submit --compute=cluster
python scripts/eval.py iclr-find-hp-cam-rff --data --submit
python scripts/eval.py iclr-find-hp-cam-rff --method --submit --compute=cluster
python scripts/eval.py iclr-find-hp-cam-rff-big-graphs --data --submit
python scripts/eval.py iclr-find-hp-cam-rff-big-graphs --method --submit --compute=cluster
```

#### Running the benchmark

```
python scripts/eval.py iclr-compare-methods --data --submit
python scripts/eval.py iclr-compare-methods --method --submit --compute=cluster
python scripts/eval.py iclr-compare-methods --summary --submit
```

```
python scripts/eval.py iclr-compare-methods-big-graphs --data --submit
python scripts/eval.py iclr-compare-methods-big-graphs --method --submit --compute=cluster
python scripts/eval.py iclr-compare-methods-big-graphs --summary --submit
```

```
python scripts/eval.py iclr-compare-methods-rff --data --submit
python scripts/eval.py iclr-compare-methods-rff --method --submit --compute=cluster
python scripts/eval.py iclr-compare-methods-rff --summary --submit
```

```
python scripts/eval.py iclr-compare-methods-rff-big-graphs --data --submit
python scripts/eval.py iclr-compare-methods-rff-big-graphs --method --submit --compute=cluster
python scripts/eval.py iclr-compare-methods-rff-big-graphs --summary --submit
```

Non-gaussian data
```
python scripts/eval.py iclr-compare-methods-non-gaussian --data --submit
python scripts/eval.py iclr-compare-methods-non-gaussian --method --submit --compute=cluster
python scripts/eval.py iclr-compare-methods-non-gaussian --summary --submit
```

### Noise transfer experiment

```
python scripts/eval.py iclr-noise-var-dist --data --submit
python scripts/eval.py iclr-noise-var-dist --summary --submit --just_data
```

```
python scripts/eval.py iclr-find-hp-notears-noise-var-transfer --data --submit
python scripts/eval.py iclr-find-hp-notears-noise-var-transfer --methods --submit --compute=cluster
python scripts/eval.py iclr-find-hp-notears-noise-var-transfer --summary --submit
```

```
python scripts/eval.py iclr-noise-var-transfer --data --submit
python scripts/eval.py iclr-noise-var-transfer --methods --submit --compute=cluster
DISABLE_NOISE_PLOT=true python scripts/eval.py iclr-noise-var-transfer --summary --submit
```

### Covariance matrices

```
python scripts/scripts/mean_absolute_covariance.py
```

```
python scripts/eval.py identifiability-trees --data --submit
COV_PLOT=true python scripts/eval.py identifiability-trees --summary --submit --just_data
```

```
python scripts/eval.py identifiability-3-node-triangle --data --submit
COV_PLOT=true python scripts/eval.py identifiability-3-node-triangle --summary --submit --just_data
```

```
python scripts/eval.py identifiability-3-nodes --data --submit
COV_PLOT=true python scripts/eval.py identifiability-3-nodes --summary --submit --just_data
```
