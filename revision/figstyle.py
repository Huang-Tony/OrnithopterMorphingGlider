"""Shared figure style for the revised manuscript: Nature-journal conventions.

Arial 5-7 pt, bold lowercase panel letters (a, b, c) without parentheses and no in-figure titles,
0.5 pt axes with outward ticks, no grid, frameless legends, 89 mm (single) / 183 mm (double) widths.
Colours follow a validated colour-blind-safe categorical palette (fixed slot order, colour follows the
entity everywhere): baseline SAC = blue, curriculum SAC = orange, residual SAC = aqua, KAN = violet;
classical controllers share neutral greys and are identified by labels.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
SINGLE = 89 * MM          # Nature single column
ONEHALF = 120 * MM
DOUBLE = 131 * MM         # revision R1: full text width of the journal template (was 183 mm, Nature double column)
TEXT = 131 * MM           # text width of the journal template (sn-jnl, 372 pt)

SLOT = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
        "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948"}
INK = "#1a1a1a"; INK2 = "#4d4d4d"; MUTED = "#8c8c8c"; LIGHT = "#d9d9d9"; GRID = "#ececec"; NEUTRAL = "#a6a6a6"
FILL_BLUE = "#dbe8f9"; FILL_GREY = "#efefef"

ENTITY = {
    "baseline": SLOT["blue"], "Baseline SAC": SLOT["blue"], "Flat SAC": SLOT["blue"],
    "curriculum": SLOT["orange"], "Curriculum SAC": SLOT["orange"],
    "residual_curriculum": SLOT["aqua"], "Residual SAC": SLOT["aqua"],
    "KAN": SLOT["violet"], "KAN student": SLOT["violet"],
    "reference": INK,
}
CLASSICAL_GREYS = ["#404040", "#5c5c5c", "#787878", "#949494", "#b0b0b0", "#c8c8c8", "#2b2b2b"]

LABELS = {"baseline": "Flat SAC", "curriculum": "Curriculum SAC", "residual_curriculum": "Residual SAC",
          "Alloc-ML": "Alloc-ML", "Heuristic": "Heuristic", "PID": "PID", "GS-PID": "GS-PID",
          "LQR": "LQR", "MPC": "MPC", "Zero": "Zero action"}


def apply():
    try:   # morphing_glider.config sets its own rcParams on import; import it first so ours win
        import morphing_glider.config  # noqa: F401
    except Exception:
        pass
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out", "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.labelpad": 2.0,
        "lines.linewidth": 1.0, "lines.markersize": 3.0, "patch.linewidth": 0.5,
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
        "ps.fonttype": 42, "figure.dpi": 150,
    })


def panel_label(ax, s, x=-0.16, y=1.02):
    ax.text(x, y, s.lower().strip("()"), transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left")


def save(fig, path_no_ext):
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    fig.savefig(path_no_ext + ".pdf"); fig.savefig(path_no_ext + ".png", dpi=600)
    plt.close(fig)
