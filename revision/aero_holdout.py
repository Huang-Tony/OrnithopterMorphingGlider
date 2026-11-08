#!/usr/bin/env python3
"""Leave-one-Reynolds-number-out check of the aerodynamic calibration (revision R1).

The nominal sectional parameters are the mean of the fits at Re = 1e5 and 2e5 (calibrate_aero_e387.py), so the
comparison with the same two data sets is partly circular. Here the parameters fitted at one Reynolds number are used
to predict the other (both directions): finite-wing lift slope (Helmbold, AR 13.3), maximum lift coefficient, stall
angle from zero lift, and minimum profile drag. The errors are compared with the half-ranges of the domain
randomization (Table tab:dr). Output: analysis/aero_holdout.json
"""

import json
import os

import common  # noqa: F401
from common import RUNS_DIR, dump_json

DR = {"cl_alpha": 0.25, "cl_max": 0.20, "alpha_stall": 0.20, "cd0": 0.35}


def main():
    d = json.load(open(os.path.join(RUNS_DIR, "analysis", "aero_validation_e387.json")))
    a, b = d["Re100000"], d["Re200000"]
    q = {"cl_alpha": ("a_3D_helmbold_per_rad",), "cl_max": ("cl_max",), "cd0": ("cd_min",)}
    out = {"doc": __doc__, "pairs": {}}
    for name, (fit, test) in {"fit 2e5 -> predict 1e5": (b, a), "fit 1e5 -> predict 2e5": (a, b)}.items():
        row = {}
        for k, (key,) in q.items():
            row[k] = float(fit[key] / test[key] - 1.0)
        st = lambda r: r["alpha_stall_drop_deg"] - r["alpha_L0_deg"]          # stall angle from zero lift
        row["alpha_stall"] = float(st(fit) / st(test) - 1.0); row["alpha_stall_deg_diff"] = float(st(fit) - st(test))
        out["pairs"][name] = row
    out["max_abs_rel_err"] = {k: max(abs(r[k]) for r in out["pairs"].values()) for k in DR}
    out["within_randomization"] = {k: out["max_abs_rel_err"][k] <= DR[k] for k in DR}
    out["dr_half_ranges"] = DR
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "aero_holdout.json"))
    print(json.dumps(out["pairs"], indent=1)); print("max", out["max_abs_rel_err"], "within DR", out["within_randomization"])


if __name__ == "__main__":
    main()
