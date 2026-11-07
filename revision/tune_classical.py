#!/usr/bin/env python3
"""Tune every classical baseline with the same budget (9 configurations) on HELD-OUT episodes
(seeds 50000.., 10 nominal + 10 fully randomised), then evaluate the selected configuration on the
standard evaluation episodes (seeds 10007.., 50 per condition) with full data capture.

Outputs: revision_runs/analysis/classical_tuning.json, revision_runs/eval_final/classical_<name>_<cond>.{json,npz}
"""

import os
import sys
import time

import numpy as np

import common  # noqa: F401
from common import RUNS_DIR, FINAL_EVAL_EPS, EVAL_SEED_BASE, CONDITIONS, dump_json
from classical import make_classical, TUNING_GRIDS
from evaluate import evaluate

TUNE_SEED_BASE = 50_000; TUNE_EPS_PER_COND = 10


def main():
    names = sys.argv[1:] or ["Zero", "Alloc-ML", "Heuristic", "PID", "GS-PID", "LQR", "MPC"]
    out_path = os.path.join(RUNS_DIR, "analysis", "classical_tuning.json")
    tuning = {}
    if os.path.exists(out_path):
        import json; tuning = json.load(open(out_path))
    for name in names:
        if name not in tuning:
            rows = []
            for cfg in TUNING_GRIDS[name]:
                vals, fails = [], []
                for cond, rs in (("nominal", 0.0), ("randomized", 1.0)):
                    res = evaluate(make_classical(name, **cfg), label=f"{name}{cfg}", condition=cond, rand_scale=rs,
                                   n_episodes=TUNE_EPS_PER_COND, seed_base=TUNE_SEED_BASE, verbose=False)
                    vals += [m["rms_yaw_horizon"] for m in res["episodes"]]; fails += [m["failure"] for m in res["episodes"]]
                rows.append({"config": cfg, "rms_h": float(np.mean(vals)), "failure": float(np.mean(fails))})
                print(f"[tune] {name:<10} {cfg} -> RMS@H={rows[-1]['rms_h']:.3f} fail={100*rows[-1]['failure']:.0f}%", flush=True)
            best = min(rows, key=lambda r: r["rms_h"])
            tuning[name] = {"grid": rows, "selected": best["config"], "selected_rms_h": best["rms_h"]}
            dump_json(tuning, out_path)
        cfg = tuning[name]["selected"]
        for cond, rs in CONDITIONS.items():
            tag = f"classical_{name}_{cond}"
            if os.path.exists(os.path.join(RUNS_DIR, "eval_final", f"{tag}.json")): continue
            evaluate(make_classical(name, **cfg), label=name, condition=cond, rand_scale=rs, n_episodes=FINAL_EVAL_EPS,
                     seed_base=EVAL_SEED_BASE, out_dir=os.path.join(RUNS_DIR, "eval_final"), tag=tag)


if __name__ == "__main__":
    main()
