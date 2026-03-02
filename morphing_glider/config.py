"""Central configuration: constants, hyperparameters, device, run-mode sizing."""

import os, sys, math, time, random, warnings, platform, hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt

# ================================================================
# GLOBAL CONFIGURATION
# ================================================================
RUN_MODE = os.environ.get("MORPHING_RUN_MODE", "medium")   # "dev", "medium", "paper"
FAST_DEV_RUN = (RUN_MODE == "dev")
MEDIUM_RUN   = (RUN_MODE == "medium")
PAPER_RUN    = (RUN_MODE == "paper")

N_TRAIN_SEEDS = 5 if PAPER_RUN else (3 if MEDIUM_RUN else 1)
EVAL_EPISODES_PER_SEED = 50 if PAPER_RUN else (20 if MEDIUM_RUN else 5)
TOTAL_TRAIN_STEPS = 1_000_000 if PAPER_RUN else (300_000 if MEDIUM_RUN else 20_000)

GLOBAL_SEED = 7
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
USE_VECNORMALIZE = True
USE_SDE = False

VECENV_MODE = "subproc"
SUBPROC_START_METHOD = "fork"

# Run flags
RUN_AERO_CALIBRATION = True
RUN_AERO_SANITY_SWEEP = True
RUN_TRAIN_BASELINE = True
RUN_TRAIN_CURRICULUM = True
RUN_TRAIN_RESIDUAL_CURRICULUM = True
RUN_FINAL_EVAL = True
RUN_ABLATION_SUITE = PAPER_RUN
RUN_DEMO_OVERLAY = True

# Gate constants
GATE_MAX_FAILURE_RATE = 0.60
GATE_MAX_FAILURE_RATE_BY_PHASE = {
    "basic_yaw": 0.75,
    "partial_twist": 0.65,
    "moderate_twist": 0.55,
    "full_twist": 0.50,
    "raw_finetune": 0.40,
}
GATE_OVERRIDE_IMPROVEMENT_THRESHOLD = 0.10
GATE_OVERRIDE_IMPROVEMENT_THRESHOLD_RESIDUAL = 0.05

MIN_EPISODE_SURVIVAL_STEPS = 20
REPLAY_RETAIN_FRACTION = 0.50

# Aero / Bezier iterations
TRAIN_AERO_PANELS = 6
EVAL_AERO_PANELS = 10
BEZIER_ITERS_TRAIN = 6
BEZIER_ITERS_EVAL = 12

# Bootstrap
BOOTSTRAP_N_PCT = 1200 if FAST_DEV_RUN else (2500 if MEDIUM_RUN else 4000)
BOOTSTRAP_N_BCA = 1500 if FAST_DEV_RUN else (3500 if MEDIUM_RUN else 6000)

# Curriculum eval padding
CURRICULUM_EVAL_RAND_PAD = 0.10

# Settling-time params
SETTLING_THRESHOLD = 0.15
SETTLING_WINDOW = 10
SETTLING_REF_MIN_ABS = 0.05
SETTLING_BAND_MIN = 0.05
SETTLING_BAND_GAIN = 0.30

# ================================================================
# CONSTANTS — geometry, reward, physics
# ================================================================
L_FIXED = 1.0          # [m] semi-span
WING_CHORD = 0.15      # [m]
DT = 0.04              # [s] timestep

DX_RANGE = (-0.50, 0.50)
DY_RANGE = (-0.20, 0.20)
DZ_RANGE = (-0.15, 0.15)

DEFAULT_YAW_TARGETS = (-0.6, -0.3, 0.0, 0.3, 0.6)
HOLD_RANGE_STEPS = (50, 80)

# Reward weights
REWARD_W_TRACK = 10.0
REWARD_W_ATT_GAIN = 0.30
REWARD_W_ATT_FLOOR = 0.08
REWARD_W_RATES_GAIN = 0.15
REWARD_W_RATES_FLOOR = 0.040
MAX_COST_ATT_REF = 1.0
MAX_COST_RATES_REF = 1.0
REWARD_W_CTRL = 0.03
REWARD_W_JERK = 0.08
REWARD_W_POWER = 0.03
REWARD_W_STRUCT = 0.08
REWARD_W_ZSYM = 0.05
REWARD_CLIP_MIN = -5.0
REWARD_CLIP_MAX = 12.0
REWARD_SURVIVAL_BONUS = 0.30
REWARD_TRACKING_SHARPNESS = 10.0
REWARD_W_WALL = 3.0
REWARD_WALL_MARGIN = 0.90

# Euler-Bernoulli / structural
EB_YOUNGS_MODULUS = 70e9
EB_SPAR_THICKNESS = 0.003

# MPC
MPC_N_HORIZON = 10
MPC_Q_R = 1.0
MPC_R_U = 0.1

# Derived wing constants
WING_AREA_TOTAL = WING_CHORD * L_FIXED * 2
WING_SPAN_TOTAL = L_FIXED * 2
ASPECT_RATIO = WING_SPAN_TOTAL / WING_CHORD
CL_ALPHA_FINITE = 2 * math.pi / (1 + 2.0 / ASPECT_RATIO)

WIND_X_ABS_MAX_MPS = 3.0

# ================================================================
# PUBLICATION SETTINGS
# ================================================================
PUBLICATION_RCPARAMS = {
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.15,
}
mpl.rcParams.update(PUBLICATION_RCPARAMS)

# ================================================================
# HYPERPARAMETER REGISTRY
# ================================================================
HYPERPARAMETER_REGISTRY: Dict[str, Any] = {
    "RUN_MODE": RUN_MODE,
    "N_TRAIN_SEEDS": N_TRAIN_SEEDS,
    "EVAL_EPISODES_PER_SEED": EVAL_EPISODES_PER_SEED,
    "TOTAL_TRAIN_STEPS": TOTAL_TRAIN_STEPS,
    "GLOBAL_SEED": GLOBAL_SEED,
    "DEVICE": str(DEVICE),
    "DT": DT,
    "L_FIXED": L_FIXED,
    "WING_CHORD": WING_CHORD,
    "DX_RANGE": DX_RANGE,
    "DY_RANGE": DY_RANGE,
    "DZ_RANGE": DZ_RANGE,
    "DEFAULT_YAW_TARGETS": DEFAULT_YAW_TARGETS,
    "REWARD_W_TRACK": REWARD_W_TRACK,
    "REWARD_W_ATT_FLOOR": REWARD_W_ATT_FLOOR,
    "REWARD_W_RATES_FLOOR": REWARD_W_RATES_FLOOR,
    "REWARD_W_CTRL": REWARD_W_CTRL,
    "REWARD_W_JERK": REWARD_W_JERK,
    "REWARD_W_POWER": REWARD_W_POWER,
    "REWARD_W_STRUCT": REWARD_W_STRUCT,
    "REWARD_W_ZSYM": REWARD_W_ZSYM,
    "REWARD_SURVIVAL_BONUS": REWARD_SURVIVAL_BONUS,
    "REWARD_TRACKING_SHARPNESS": REWARD_TRACKING_SHARPNESS,
    "USE_VECNORMALIZE": USE_VECNORMALIZE,
    "USE_SDE": USE_SDE,
    "VECENV_MODE": VECENV_MODE,
    "TRAIN_AERO_PANELS": TRAIN_AERO_PANELS,
    "EVAL_AERO_PANELS": EVAL_AERO_PANELS,
    "BOOTSTRAP_N_BCA": BOOTSTRAP_N_BCA,
    "GATE_MAX_FAILURE_RATE": GATE_MAX_FAILURE_RATE,
}


# ================================================================
# REPRODUCIBILITY
# ================================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ================================================================
# PLOTTING HELPERS (used widely)
# ================================================================
def _add_panel_label(ax, label, x=-0.08, y=1.05, fontsize=14, fontweight="bold"):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight=fontweight, va="bottom", ha="right")


def _save_fig(fig, path, caption="", dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[SAVED] {path}" + (f"  # {caption}" if caption else ""))
