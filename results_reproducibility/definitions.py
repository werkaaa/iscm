from pathlib import Path

# directories
ROOT_DIR = Path(__file__).parents[0]
PROJECT_DIR = Path(__file__).parents[1]
SCRATCH=None

# subdirectory names
EXPERIMENTS_SUBDIR = "experiment_configs"
RESULTS_SUBDIR = "results"
SCHEMAS_SUBDIR = "input_graph_schemas"

EXPERIMENT_CONFIG_DATA = "data.yaml"
EXPERIMENT_CONFIG_METHODS = "methods.yaml"

# experiments
EXPERIMENT_DATA = "data"
EXPERIMENT_PREDS = "predictions"
EXPERIMENT_SUMMARY = "summary"

FILE_DATA_G = "g.csv"
FILE_DATA_SAMPLES = "x.csv"
FILE_DATA_PRESTAND_VARS = "prestand_vars.csv"
FILE_DATA_NOISE_VARS = "noise_vars.csv"
FILE_DATA_META = f"info.json"

# rng entropies; these integers must be different to guarantee different randomness in data during train, val, test
RNG_ENTROPY_EVAL = 0
RNG_ENTROPY_HPARAMS = 100
EVAL_MODE = "eval"
HPARAMS_MODE = "hp-tuning"

# data generation methods
SCM = 'raw'
iSCM = 'alternate'
sSCM = 'final'

# data generation heuristics
SQUIRES = 'squires'
MOOIJ = 'mooij'

# graphs
USF = 'USF'
SFT = 'SFT'
SF = 'SF'
UNIFORM = 'uniform'
ER = 'ER'
SCHEMA = 'schema'

# causal graph discovery methods
NOTEARS = 'notears'
AVICI = 'avici'
PC = 'pc'
GES = 'ges'
GOLEM_EV = 'golem_ev'
GOLEM_NV = 'golem_nv'
GOLEM = 'golem'
LINGAM = 'lingam'
SCORE = 'score'
CAM = 'cam'

VAR_SORT_REGRESS = 'varsortregress'
R2_SORT_REGRESS = 'r2sortregress'
RANDOM_SORT_REGRESS = 'randomsortregress'

# sortability methods
R2_SORTABILITY = "r2_sortability"
CEV_SORTABILITY = "cev_sortability"
INDG_SORTABILITY = "indg_sortability"
NS_SORTABILITY = "ns_sortability"
MBS_SORTABILITY = "mbs_sortability"
VAR_SORTABILITY = "var_sortability"

# avici checkpoints
LINEAR = 'linear'
RFF = 'rff'
AVICI_CHECKPOINT_LINEAR = "cache/avici/linear/checkpoint"
AVICI_CHECKPOINT_RFF = "cache/avici/rff/checkpoint"

# yaml special
YAML_RUN = "__run__"

DEFAULT_RUN_KWARGS = {"n_cpus": 1, "n_gpus": 0, "length": "short"}

# plotting configuration
COV_PLOT = 'COV_PLOT'
DISABLE_NOISE_PLOT = 'DISABLE_NOISE_PLOT'
