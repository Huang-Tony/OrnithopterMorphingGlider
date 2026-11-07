"""Shared settings/helpers for the revision-R1 staged pipeline.

Import this module BEFORE anything from `morphing_glider`: it pins RUN_MODE=paper so that config-level
constants (bootstrap sizes, eval episodes, ...) match the paper-mode run.
"""

import json
import os
import subprocess
import sys
import time

os.environ.setdefault("MORPHING_RUN_MODE", "paper")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

RUNS_DIR = os.environ.get("REVISION_RUNS_DIR", os.path.join(REPO, "revision_runs"))

# ---- Paper-mode sizing (identical to run_simulation.py, RUN_MODE='paper') ----
N_ENVS = 10
MAX_STEPS_EP = 200
EVAL_EVERY = 50_000          # training-time evaluation interval
TRAIN_EVAL_EPS = 25          # episodes per training-time evaluation
FINAL_EVAL_EPS = 50          # episodes per seed per condition in the final evaluation
GLOBAL_SEED = 7
EVAL_SEED_BASE = GLOBAL_SEED + 10000
TOTAL_STEPS = 1_000_000      # revision R1: identical budget for baseline, curriculum and residual SAC
TRAIN_SEEDS = [GLOBAL_SEED + 100 * s for s in range(5)]          # 7, 107, 207, 307, 407
ALGO_SEED_OFFSET = {"baseline": 0, "curriculum": 10, "residual_curriculum": 20}

# ---- Common evaluation protocol (revision R1: one 65 deg limit for every controller) ----
EVAL_RPL_DEG = 65.0
EVAL_COUPLING = 1.0
EVAL_STAB_W = 0.03
CONDITIONS = {"nominal": 0.0, "randomized": 1.0}


def algo_seed(algo: str, base_seed: int) -> int:
    return int(base_seed) + ALGO_SEED_OFFSET[algo]


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", REPO, "diff", "--quiet", "--", "morphing_glider", "revision"]) != 0
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def _to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _to_jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if callable(o):
        return getattr(o, "__name__", repr(o))
    return o


def dump_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_to_jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def stamp() -> dict:
    import platform
    import torch
    from morphing_glider.config import HYPERPARAMETER_REGISTRY, DEVICE
    return {"git_commit": git_commit(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python": sys.version.split()[0], "torch": torch.__version__, "device": str(DEVICE),
            "platform": platform.platform(), "hyperparameters": dict(HYPERPARAMETER_REGISTRY)}
