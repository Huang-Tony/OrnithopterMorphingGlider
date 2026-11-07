#!/usr/bin/env python3
"""Two-stage tuning of the classical baselines (revision R1, airframe v3; rule fixed 2026-11-07 before any v3
comparison with SAC was seen).

Stage 1  the 9-configuration grid of classical.TUNING_GRIDS (as tune_classical.py).
Stage 2  local refinement from the best grid point: Nelder-Mead in log-parameter space (initial simplex = factor 2
         per parameter, so the search can leave the grid), 30 further objective evaluations, parameters clipped to
         physical bounds. Total budget: 39 evaluations for every parametric controller.
Objective (both stages): mean RMS@H over the 20 held-out tuning episodes (seeds 50000.., 10 nominal + 10 fully
randomized), disjoint from the evaluation seeds. The refinement gain reported in the paper is
(f_stage1 - f_stage2) / f_stage1; a small gain indicates that the tuning has converged.
The selected configuration is then evaluated once on the standard evaluation episodes (seeds 10007.., 50 per
condition). Outputs: analysis/tuning/<name>.json, analysis/classical_tuning.json, eval_final/classical_<name>_<cond>.*,
analysis/TUNING_DONE (sentinel for auto_pipeline.sh).

Usage: refine_classical.py --all [--workers 4]     or     refine_classical.py <name>
"""

import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402
from common import RUNS_DIR, FINAL_EVAL_EPS, EVAL_SEED_BASE, CONDITIONS, dump_json  # noqa: E402
import numpy as np  # noqa: E402

TUNE_SEED_BASE = 50_000; TUNE_EPS_PER_COND = 10; N_REFINE = 30
BOUNDS = {"scale": (0.03, 0.5), "gain": (0.05, 10.0), "x_range": (0.03, 0.5), "roll_gain": (0.01, 3.0),
          "k_att": (0.1, 6.0), "bank_max_deg": (5.0, 60.0), "k_yaw": (0.1, 6.0)}
PARAMETRIC = ["Heuristic", "PID", "GS-PID", "LQR", "MPC", "Alloc-ML"]
TDIR = os.path.join(RUNS_DIR, "analysis", "tuning")


def objective(name, cfg):
    from evaluate import evaluate
    from classical import make_classical
    vals, fails = [], []
    for cond, rs in (("nominal", 0.0), ("randomized", 1.0)):
        res = evaluate(make_classical(name, **cfg), label=f"{name}", condition=cond, rand_scale=rs,
                       n_episodes=TUNE_EPS_PER_COND, seed_base=TUNE_SEED_BASE, verbose=False)
        vals += [m["rms_yaw_horizon"] for m in res["episodes"]]; fails += [m["failure"] for m in res["episodes"]]
    return float(np.mean(vals)), float(np.mean(fails))


def tune_one(name):
    import torch; torch.set_num_threads(1)
    from classical import TUNING_GRIDS
    from scipy.optimize import minimize
    path = os.path.join(TDIR, f"{name}.json")
    if os.path.exists(path): return name
    os.makedirs(TDIR, exist_ok=True)
    old = os.path.join(RUNS_DIR, "analysis", "classical_tuning.json")
    old = json.load(open(old)).get(name) if os.path.exists(old) else None
    if old and old.get("grid") and [r["config"] for r in old["grid"]] == TUNING_GRIDS[name]:
        grid = old["grid"]                                   # stage 1 already computed with the same grid
    else:
        grid = []
        for cfg in TUNING_GRIDS[name]:
            f, fl = objective(name, cfg); grid.append({"config": cfg, "rms_h": f, "failure": fl})
            print(f"[stage1] {name:<10} {cfg} -> {f:.3f} fail {100*fl:.0f}%", flush=True)
    best1 = min(grid, key=lambda r: r["rms_h"])
    keys = sorted(best1["config"]); lo = np.log([BOUNDS[k][0] for k in keys]); hi = np.log([BOUNDS[k][1] for k in keys])
    x0 = np.log([best1["config"][k] for k in keys])
    trace, cache = [], {}

    def f(x):
        x = np.clip(x, lo, hi); key = tuple(np.round(x, 4))
        if key not in cache:
            cfg = {k: float(np.exp(v)) for k, v in zip(keys, x)}
            val, fl = objective(name, cfg); cache[key] = val
            trace.append({"config": cfg, "rms_h": val, "failure": fl})
            print(f"[stage2] {name:<10} {({k: round(v, 3) for k, v in cfg.items()})} -> {val:.3f} fail {100*fl:.0f}%", flush=True)
        return cache[key]
    simplex = np.vstack([x0] + [x0 + math.log(2.0) * np.eye(len(keys))[i] for i in range(len(keys))])
    minimize(f, x0, method="Nelder-Mead", options={"maxfev": N_REFINE, "initial_simplex": simplex,
                                                    "xatol": 0.02, "fatol": 1e-4})
    best2 = min(trace, key=lambda r: r["rms_h"]) if trace else best1
    sel = best2 if best2["rms_h"] < best1["rms_h"] else best1
    on_edge = {k: (best1["config"][k] in (min(c["config"][k] for c in grid), max(c["config"][k] for c in grid))) for k in keys}
    out = {"grid": grid, "refinement": trace, "stage1_best": best1, "stage2_best": best2, "selected": sel["config"],
           "selected_rms_h": sel["rms_h"], "refinement_gain": (best1["rms_h"] - sel["rms_h"]) / best1["rms_h"],
           "n_evaluations": len(grid) + len(trace), "stage1_best_on_grid_edge": on_edge}
    dump_json(out, path)
    print(f"[done] {name}: stage1 {best1['rms_h']:.3f} -> selected {sel['rms_h']:.3f} "
          f"(gain {100*out['refinement_gain']:.1f}%) {sel['config']}", flush=True)
    return name


def eval_one(name, cfg):
    import torch; torch.set_num_threads(1)
    from evaluate import evaluate
    from classical import make_classical
    for cond, rs in CONDITIONS.items():
        evaluate(make_classical(name, **cfg), label=name, condition=cond, rand_scale=rs, n_episodes=FINAL_EVAL_EPS,
                 seed_base=EVAL_SEED_BASE, out_dir=os.path.join(RUNS_DIR, "eval_final"), tag=f"classical_{name}_{cond}",
                 verbose=False)
    return name


def run_all(workers):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(tune_one, PARAMETRIC))
    merged_path = os.path.join(RUNS_DIR, "analysis", "classical_tuning.json")
    merged = json.load(open(merged_path)) if os.path.exists(merged_path) else {}
    merged.setdefault("Zero", {"grid": [], "selected": {}, "selected_rms_h": None})
    for n in PARAMETRIC:
        merged[n] = json.load(open(os.path.join(TDIR, f"{n}.json")))
    dump_json(merged, merged_path)
    for n in ["Zero"] + PARAMETRIC:                        # final evaluation of the selected configurations
        for p in glob.glob(os.path.join(RUNS_DIR, "eval_final", f"classical_{n}_*")): os.remove(p)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(eval_one, ["Zero"] + PARAMETRIC, [merged["Zero"]["selected"]] + [merged[n]["selected"] for n in PARAMETRIC]))
    open(os.path.join(RUNS_DIR, "analysis", "TUNING_DONE"), "w").write("ok\n")
    print("[refine] all done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("name", nargs="?"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    run_all(a.workers) if a.all else tune_one(a.name)
