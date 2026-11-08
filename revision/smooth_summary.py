#!/usr/bin/env python3
"""Summary of the smoothness fine-tuning study (revision R1): SAC-S vs flat SAC, paired by seed and episode.

Both are evaluated on the identical evaluation episodes (EVAL_SEED_BASE, 50 per seed and condition) and, for the
measured wing mode, on the identical robustness episodes (seed base 30000, 20 per seed). For each metric: the mean
of the seed means of each policy, the per-seed difference SAC-S minus SAC (min, max, mean) and a hierarchical
bootstrap 95% CI of the mean difference (resample seeds, then episodes within seed, paired by episode).
Inputs: analysis/smooth_study.json, eval_final/sac_baseline_seed<s>_<cond>.json,
eval_robust/sac_baseline_seed<s>_struct_mode.json. Output: analysis/smooth_summary.json
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from common import RUNS_DIR, dump_json  # noqa: E402
import numpy as np  # noqa: E402

SEEDS = [7, 107, 207, 307, 407]
METRICS = ("rms_yaw_horizon", "failure", "hf_action_power_frac_2Hz", "mean_tip_speed_mps", "mean_sq_action_increment",
           "rms_roll_deg")


def flat(seed, cond):
    p = (os.path.join(RUNS_DIR, "eval_robust", f"sac_baseline_seed{seed}_struct_mode.json") if cond == "struct_mode"
         else os.path.join(RUNS_DIR, "eval_final", f"sac_baseline_seed{seed}_{cond}.json"))
    if not os.path.exists(p): return None
    E = json.load(open(p))["episodes"]
    return {k: [float(e[k]) for e in E] for k in METRICS}


def hboot(diffs, n=4000, rng=np.random.default_rng(0)):
    """diffs: list (seeds) of arrays (episodes) of paired differences."""
    out = []
    for _ in range(n):
        s = rng.integers(0, len(diffs), len(diffs))
        out.append(np.mean([np.mean(diffs[i][rng.integers(0, len(diffs[i]), len(diffs[i]))]) for i in s]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    st = json.load(open(os.path.join(RUNS_DIR, "analysis", "smooth_study.json")))
    out = {"doc": __doc__, "selected": st["selected"], "conditions": {}}
    for cond in ("nominal", "randomized", "struct_mode"):
        row = {}
        for k in METRICS:
            a, b, d = [], [], []
            for s in SEEDS:
                sm = st.get("evaluation", {}).get(f"{s}|{cond}"); fl = flat(s, cond)
                if not sm or not fl: continue
                x, y = np.asarray(sm[k]), np.asarray(fl[k])
                a.append(x.mean()); b.append(y.mean())
                if len(x) == len(y): d.append(x - y)
            if not a: continue
            row[k] = {"smooth": float(np.mean(a)), "flat": float(np.mean(b)), "smooth_seed": [float(v) for v in a],
                      "flat_seed": [float(v) for v in b]}
            if d:
                sd = [float(np.mean(v)) for v in d]
                row[k].update(diff_mean=float(np.mean(sd)), diff_seed_min=min(sd), diff_seed_max=max(sd), diff_ci=hboot(d),
                              n_seeds_lower=int(sum(v < 0 for v in sd)))
        out["conditions"][cond] = row
    pil = st.get("pilot", {})
    if pil:
        out["pilot"] = pil
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "smooth_summary.json"))
    for cond, row in out["conditions"].items():
        for k, v in row.items():
            print(f"{cond:11s} {k:26s} SAC-S {v['smooth']:.3f}  SAC {v['flat']:.3f}  diff {v.get('diff_mean', float('nan')):+.3f} "
                  f"CI {v.get('diff_ci')}  seeds lower {v.get('n_seeds_lower')}")


if __name__ == "__main__":
    main()
