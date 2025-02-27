import matplotlib.font_manager  # avoids loading error of fontfamily["serif"]
import matplotlib as mpl

COLOR_SATURATION = 0.8
DPI = 300

COLORS = [
    "#1f77b4",  # 0 blue
    "#ff7f0e",  # 1 orange
    "#2ca02c",  # 2 green
    '#d62728',  # 3 red
    '#9467bd',  # 4 purple
]

METRICS_HIGHER_PREC = [
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "ap",
    "ece",
    "brier",
]

DATA_GEN_METHODS_ORDERING = ["SCM", "Standardized SCM", "iSCM"]
DATA_GEN_METHODS_ORDERING_SHORT_STAND = ["SCM", "Stand. SCM", "iSCM"]
DATA_GEN_METHODS_ORDERING_SHORT = ["Standardized SCM", "iSCM"]
HEURISTIC_DATA_GEN_METHODS_ORDERING = ["Mooij", "Mooij + noise scaling", "Squires"]
DATA_GEN_METHODS_CONFIG = {
    "SCM": COLORS[0],
    "Standardized SCM": COLORS[1],
    "iSCM": COLORS[2]
}

HEURISTIC_DATA_GEN_METHODS_CONFIG = {
    "Mooij": COLORS[3], # red
    "Squires": COLORS[4] # purple
}

OUTLIER_SIZE = 0.8
WEIGHT_RANGE_TITLES = ['[0.3, 0.8]', '[1.3, 3.0]']

NEURIPS_LINE_WIDTH = 5.5  # Text width: 5.5in (double figure minus spacing 0.2in).
FIG_SIZE_NEURIPS_TRIPLE = (NEURIPS_LINE_WIDTH / 3, NEURIPS_LINE_WIDTH / 3 * 4 / 6)
FIG_SIZE_NEURIPS_DOUBLE = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 4 / 6)
FIG_SIZE_NEURIPS_DOUBLE_SHORT = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 3 / 6)
FIG_SIZE_NEURIPS_SINGLE = (NEURIPS_LINE_WIDTH, NEURIPS_LINE_WIDTH * 4 / 6)
FIG_SIZE_NEURIPS_ONE_AND_HALF = (NEURIPS_LINE_WIDTH / 1.5, NEURIPS_LINE_WIDTH / 1.5 * 4 / 6)

VERTICAL_SCALING_COMP_TOP = 6 / 5
VERTICAL_SCALING_COMP_BOTTOM = (1 + 3.5 / 14.5) * 6 / 5

NEURIPS_RCPARAMS = {
    "figure.autolayout": False,  # `False` makes `fig.tight_layout()` not work
    "figure.figsize": FIG_SIZE_NEURIPS_ONE_AND_HALF,
    # "figure.dpi": DPI,             # messes up figisize
    # Axes params
    "axes.linewidth": 0.5,  # Matplotlib's current default is 0.8.
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,

    "hatch.linewidth": 0.3,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    'xtick.major.pad': 3.0,
    'xtick.major.size': 1.75,
    'xtick.minor.pad': 1.0,
    'xtick.minor.size': 1.0,

    'ytick.major.pad': 1.0,
    'ytick.major.size': 1.75,
    'ytick.minor.pad': 1.0,
    'ytick.minor.size': 1.0,

    "axes.labelpad": 0.5,
    # Grid
    "grid.linewidth": 0.3,
    # Plot params
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    'errorbar.capsize': 3.0,
    # Font
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",  # use serif rather than sans-serif
    "font.serif": "Times New Roman",  # use "Times New Roman" as the standard font
    "font.size": 8.5,
    "axes.titlesize": 8.5,  # LaTeX default is 10pt font.
    "axes.labelsize": 8.5,  # LaTeX default is 10pt font.
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Legend
    "legend.fontsize": 7,  # Make the legend/label fonts a little smaller
    "legend.frameon": True,  # Remove the black frame around the legend
    "legend.handletextpad": 0.3,
    "legend.borderaxespad": 0.2,
    "legend.labelspacing": 0.1,
    "patch.linewidth": 0.5,
    # PDF
    "pgf.texsystem": "xelatex",  # use Xelatex which is TTF font aware
    "pgf.rcfonts": False,  # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": (
        r'\usepackage{fontspec}'
        r'\usepackage{unicode-math}'
        r'\setmainfont{Times New Roman}'
    ),
    "text.latex.preamble": (
        r'\usepackage{textcomp}'
        r'\usepackage{amsmath}'
        r'\usepackage{amssymb}'
        r'\usepackage{amsfonts}'
    ),
}

NEURIPS_RCPARAMS_SHORT = {
    # # Font
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",  # use serif rather than sans-serif
    "font.serif": "Times New Roman",  # use "Times New Roman" as the standard font

    # PDF
    "pgf.texsystem": "xelatex",  # use Xelatex which is TTF font aware
    "pgf.rcfonts": False,  # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": (
        r'\usepackage{fontspec}'
        r'\usepackage{unicode-math}'
        r'\setmainfont{Times New Roman}'
    ),
    "text.latex.preamble": (
        r'\usepackage{textcomp}'
        r'\usepackage{amsmath}'
        r'\usepackage{amssymb}'
        r'\usepackage{amsfonts}'
    ),
}

mpl.rcParams.update(NEURIPS_RCPARAMS)
