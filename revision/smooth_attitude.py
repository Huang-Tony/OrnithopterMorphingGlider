#!/usr/bin/env python3
"""Attitude of SAC-S vs flat SAC (revision R1, smoothness study): RMS and maximum pitch and roll angles and rates on
the identical nominal evaluation episodes (EVAL_SEED_BASE, 50 per seed). The smoothness study itself recorded only
tracking and actuation metrics. Output: analysis/smooth_attitude.json
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from common import RUNS_DIR, EVAL_SEED_BASE, FINAL_EVAL_EPS, dump_json  # noqa: E402
import numpy as np  # noqa: E402
from evaluate import evaluate, load_sac  # noqa: E402

K = ("rms_pitch_deg", "max_abs_pitch_deg", "rms_pitch_rate_dps", "rms_roll_rate_dps", "rms_roll_deg", "max_abs_roll_deg")


def main():
    sel = json.load(open(os.path.join(RUNS_DIR, "analysis", "smooth_study.json")))["selected"]
    out = {"doc": __doc__, "seeds": {}}
    for s in (7, 107, 207, 307, 407):
        E = evaluate(load_sac(os.path.join(RUNS_DIR, "smooth", "models", f"smooth_seed{s}_{sel}.zip")), label="sac_s", condition="nominal",
                     rand_scale=0.0, n_episodes=FINAL_EVAL_EPS, seed_base=EVAL_SEED_BASE, verbose=False)["episodes"]
        F = json.load(open(os.path.join(RUNS_DIR, "eval_final", f"sac_baseline_seed{s}_nominal.json")))["episodes"]
        out["seeds"][str(s)] = {k: {"smooth": float(np.mean([e[k] for e in E])), "flat": float(np.mean([e[k] for e in F]))} for k in K}
    out["mean"] = {k: {w: float(np.mean([v[k][w] for v in out["seeds"].values()])) for w in ("smooth", "flat")} for k in K}
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "smooth_attitude.json")); print(json.dumps(out["mean"], indent=1))


if __name__ == "__main__":
    main()
