"""Shared figure style for the revised manuscript (print, light surface).

Colours follow a validated categorical palette (fixed slot order, colour follows the entity everywhere):
baseline SAC = slot 1 blue, curriculum SAC = slot 2 orange, residual SAC = slot 3 aqua, KAN student =
slot 7 violet; classical controllers share neutral greys (they are identified by axis labels/legends).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

SLOT = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
        "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948"}
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8c8b86"; GRID = "#e4e3df"; NEUTRAL = "#a8a7a1"

ENTITY = {
    "baseline": SLOT["blue"], "Baseline SAC": SLOT["blue"],
    "curriculum": SLOT["orange"], "Curriculum SAC": SLOT["orange"],
    "residual_curriculum": SLOT["aqua"], "Residual SAC": SLOT["aqua"],
    "KAN": SLOT["violet"], "KAN student": SLOT["violet"],
    "reference": INK,
}
CLASSICAL_GREYS = ["#4f4e4a", "#6b6a65", "#85847e", "#9f9e98", "#b9b8b2", "#cfcec8", "#3b3a37"]

LABELS = {"baseline": "Baseline SAC", "curriculum": "Curriculum SAC", "residual_curriculum": "Residual SAC",
          "Alloc-ML": "Alloc-ML", "Heuristic": "Heuristic", "PID": "PID", "GS-PID": "GS-PID",
          "LQR": "LQR", "MPC": "MPC", "Zero": "Zero action"}


def apply():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
        "lines.markersize": 4.5, "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "figure.dpi": 150,
    })


def panel_label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, path_no_ext):
    import os
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    fig.savefig(path_no_ext + ".pdf"); fig.savefig(path_no_ext + ".png", dpi=400)
    plt.close(fig)
