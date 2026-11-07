#!/usr/bin/env python3
"""Calibrate the strip-theory aerodynamic proxy to public wind-tunnel data (revision R1, model v2).

Data: Eppler 387, NASA Langley LTPT, NASA TM-4062 Table B1, Re = 1e5 and 2e5 (bracketing the glider's
Re ~ 1.5e5). All nominal parameters below are derived from the data (mean of the two Reynolds numbers):
  * strip lift slope   = finite-wing slope (Helmbold, AR = 13.3) built from the measured 2-D slope
                         (the proxy has no explicit downwash model, so the strip slope carries the 3-D effect);
  * cl_max, saturation sharpness n of  c_l = a*al / (1 + |a*al/cl_max|^n)^(1/n)  (fitted pre-stall);
  * stall angle (from zero lift) and post-stall lift ratio (measured abrupt drop);
  * profile drag cd_p = cdp0 + kp*cl^2 (fitted), plus ideal induced drag cl^2/(pi e AR) with e = 0.9
    and a parasitic allowance d_cd0 = 0.005 for fuselage/joint interference (assumption, randomised);
  * camber offset alpha0 re-trimmed so that lift = weight at zero geometric AoA and V0 (as in v1).
Writes revision/data/aero_calibration_e387.json; NOMINAL_PHYS (domain_randomizer.py) is set from it.
"""

import json
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares, brentq

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: F401,E402
from validate_aero_e387 import load_e387, helmbold, AR  # noqa: E402

E_OSWALD = 0.90; D_CD0_PARASITIC = 0.005


def sat(x, cl_max, n):
    return x / (1.0 + np.abs(x / cl_max) ** n) ** (1.0 / n)


def main():
    E = load_e387(); per = {}
    for Re, d in E.items():
        a, cl, cd = d[:, 0], d[:, 1], d[:, 2]
        lin = (a >= -3.0) & (a <= 6.1)
        p = np.polyfit(np.radians(a[lin]), cl[lin], 1); a0, aL0 = float(p[0]), float(-p[1] / p[0])
        ap = np.radians(a) - aL0                                   # angle from zero lift [rad]
        i_max = int(np.argmax(cl)); pre = ap <= ap[i_max] + 1e-9
        r = least_squares(lambda z: sat(a0 * ap[pre], z[0], z[1]) - cl[pre], x0=[1.25, 4.0],
                          bounds=([0.8, 1.0], [2.0, 30.0]))
        post = np.where((a > a[i_max]) & (cl < 0.85 * cl[i_max]))[0]
        ok = np.isfinite(cd) & (cl <= 1.1) & (cl >= 0.0)
        kp, cdp0 = np.polyfit(cl[ok] ** 2, cd[ok], 1)
        per[Re] = {"a0_2D": a0, "a_3D_helmbold": helmbold(a0), "alpha_L0_rad": aL0, "cl_max_fit": float(r.x[0]),
                   "cl_max_meas": float(cl[i_max]), "sat_n": float(r.x[1]), "fit_rmse": float(np.sqrt(np.mean(r.fun ** 2))),
                   "alpha_stall_from_L0_rad": float(ap[post[0]]) if post.size else float("nan"),
                   "post_stall_ratio": float(cl[post[0]] / cl[i_max]) if post.size else float("nan"),
                   "cdp0": float(cdp0), "kp": float(kp)}
    m = lambda k: float(np.nanmean([v[k] for v in per.values()]))
    k_ideal = 1.0 / (math.pi * E_OSWALD * AR)
    cal = {"cl_alpha": m("a_3D_helmbold"), "lift_scale": 1.0, "cl_max": m("cl_max_meas"), "cl_sat_n": m("sat_n"),
           "alpha_stall": m("alpha_stall_from_L0_rad"), "post_stall_cl_factor": m("post_stall_ratio"),
           "cd0": m("cdp0") + D_CD0_PARASITIC, "k_induced": m("kp") + k_ideal}
    # re-trim alpha0 (lift = weight at zero geometric AoA, V0) with the gull-arched rest shape
    from morphing_glider.physics.domain_randomizer import NOMINAL_PHYS
    import identify_plant as ip
    ip.PHYS.clear(); ip.PHYS.update(dict(NOMINAL_PHYS)); ip.PHYS.update(cal)
    W = ip.PHYS["mass"] * ip.PHYS["g"]
    cal["alpha0"] = brentq(lambda a0: (ip.PHYS.__setitem__("alpha0", a0), ip.loads(np.zeros(6))[2] - W)[1], -0.05, 0.2)
    F0, M0, L0, D0 = ip.loads(np.zeros(6))
    out = {"per_reynolds": {str(k): v for k, v in per.items()}, "calibrated": cal, "e_oswald": E_OSWALD,
           "d_cd0_parasitic": D_CD0_PARASITIC, "k_ideal_induced": k_ideal, "AR": AR,
           "trim_check": {"lift_N": float(L0), "weight_N": float(W), "drag_N": float(D0), "L_over_D": float(L0 / D0)},
           "v1_values": {k: NOMINAL_PHYS[k] for k in ("cl_alpha", "lift_scale", "cl_max", "alpha_stall", "cd0",
                                                       "k_induced", "alpha0")}}
    path = os.path.join(HERE, "data", "aero_calibration_e387.json")
    json.dump(out, open(path, "w"), indent=1)
    for Re, v in per.items():
        print(f"Re={Re}: n={v['sat_n']:.2f} clmax_fit={v['cl_max_fit']:.3f} rmse={v['fit_rmse']:.3f} stall={math.degrees(v['alpha_stall_from_L0_rad']):.1f}deg "
              f"post={v['post_stall_ratio']:.2f} cdp0={v['cdp0']:.4f} kp={v['kp']:.4f}")
    print("calibrated:", {k: round(v, 4) for k, v in cal.items()})
    print("trim:", {k: round(v, 3) for k, v in out["trim_check"].items()}, "->", path)


if __name__ == "__main__":
    main()
