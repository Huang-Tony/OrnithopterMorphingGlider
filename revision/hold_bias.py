#!/usr/bin/env python3
"""Steady yaw-rate offset of flat SAC in straight-flight (hold-heading, r_ref = 0) segments (revision R1, Discussion).

For every constant-reference segment of at least 1.6 s in the nominal evaluation episodes (5 seeds x 50), the mean
yaw-rate error after the transient (first 1 s after a switch, 0.6 s at the episode start) is the segment's steady
offset. Reported: the median absolute offset over the r_ref = 0 segments, its 90th percentile, and per-seed values.
Output: analysis/hold_bias.json
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from common import RUNS_DIR, TRAIN_SEEDS, dump_json  # noqa: E402
import numpy as np  # noqa: E402


def offsets(npz):
    a = np.load(npz); out = []
    for rr, r in zip(a["yaw_ref"], a["yaw_rate"]):
        n = int(np.isfinite(r).sum()); x, y = rr[:n], r[:n]
        b = [0] + [k for k in range(1, n) if x[k] != x[k - 1]] + [n]
        for k in range(len(b) - 1):
            i0, i1 = b[k], b[k + 1]
            if i1 - i0 < 40: continue
            st = i0 + (15 if k == 0 else 25)
            out.append((float(x[i0]), float(np.mean(y[st:i1] - x[i0]))))
    return out


def main():
    res = {"doc": __doc__, "seeds": {}}; allz = []
    for s in TRAIN_SEEDS:
        z = [b for ref, b in offsets(os.path.join(RUNS_DIR, "eval_final", f"sac_baseline_seed{s}_nominal.npz")) if ref == 0.0]
        res["seeds"][str(s)] = {"n": len(z), "mean": float(np.mean(z)), "median_abs": float(np.median(np.abs(z)))}; allz += z
    res["median_abs"] = float(np.median(np.abs(allz))); res["p90_abs"] = float(np.percentile(np.abs(allz), 90)); res["n"] = len(allz)
    dump_json(res, os.path.join(RUNS_DIR, "analysis", "hold_bias.json")); print(res["median_abs"], res["p90_abs"], res["n"])


if __name__ == "__main__":
    main()
