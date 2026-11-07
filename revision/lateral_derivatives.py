#!/usr/bin/env python3
"""Shape dependence of the lateral-directional derivatives of the airframe-v3 model (revision R1).

Why: every classical baseline is designed about the neutral wing shape. This script quantifies how strongly the
derivatives that govern lateral stability change with the wing shape, which explains the behaviour of the
classical controllers (Sect. Results) and the shape strategy of the learned policy.

Quantities (central differences of the aerodynamic model at V0, alpha_body = 0, omega = 0, nominal physics):
  L_beta, N_beta, Y_beta  [N*m/rad, N*m/rad, N/rad]   roll / yaw moment and side force per unit sideslip;
  Cl_beta, Cn_beta         the same, non-dimensional, converted to the conventional NED axes
                           (Cl_beta_NED < 0: stable dihedral effect; Cn_beta_NED > 0: weathercock stable);
  N_ds  [N*m/m]            yaw effectiveness of differential sweep;   L_dy [N*m/m] roll effectiveness of
                           differential span;   L_dd [N*m/m] roll effectiveness of differential dihedral;
  N_beta one-sided secants (beta in [-8, 0] deg and [0, +8] deg) under differential span / sweep.
Body frame of the simulator: x forward, y towards the 'R' (+y) wing, z up; NED conversion: L_NED = L,
N_NED = -N, beta_NED = -beta, hence Cl_beta_NED = -L_beta/(q S b), Cn_beta_NED = N_beta/(q S b).
Output: revision_runs/analysis/lateral_derivatives.json
"""

import math
import os

import common  # noqa: F401
from common import RUNS_DIR, dump_json
import numpy as np
import identify_plant as ip

V0 = ip.V0
S_REF, B_REF = 0.30, 2.0           # wing reference area [m^2] and span [m] (as in identify_plant)
QSB = 0.5 * ip.PHYS["rho"] * V0 ** 2 * S_REF * B_REF
U = {k: np.asarray(v, float) for k, v in ip.VIRTUAL.items()}


def loads_beta(a, beta_deg):
    b = math.radians(beta_deg)
    v = V0 * np.array([math.cos(b), math.sin(b), 0.0])
    F, M, L, D = ip.loads(a, v_body=v)
    return F, M


def arch_height(a):
    sR = ip.spar([a[0], ip.L_FIXED + a[1], a[2]])
    return float(max(sR.p1[2], sR.p2[2]))


def derivs(a, db=4.0):
    Fp, Mp = loads_beta(a, +db); Fm, Mm = loads_beta(a, -db); r = 2 * math.radians(db)
    Lb, Nb, Yb = (Mp[0] - Mm[0]) / r, (Mp[2] - Mm[2]) / r, (Fp[1] - Fm[1]) / r
    eff = {}
    for name, key, idx, h in (("N_ds", "diff_sweep", 2, 0.05), ("L_dy", "diff_span", 0, 0.02),
                              ("L_dd", "diff_dihedral", 0, 0.02)):
        eff[name] = float((ip.loads(a + h * U[key])[1][idx] - ip.loads(a - h * U[key])[1][idx]) / (2 * h))
    F0, M0, L0, D0 = ip.loads(a)
    return {"L_beta": float(Lb), "N_beta": float(Nb), "Y_beta": float(Yb),
            "Cl_beta_NED": float(-Lb / QSB), "Cn_beta_NED": float(Nb / QSB), **eff,
            "lift_N": float(L0), "My0": float(M0[1]), "arch_height_m": arch_height(a)}


def one_sided_Nbeta(a, db=8.0):
    m0 = loads_beta(a, 0.0)[1][2]; r = math.radians(db)
    return {"N_beta_neg": float((m0 - loads_beta(a, -db)[1][2]) / r),
            "N_beta_pos": float((loads_beta(a, +db)[1][2] - m0) / r)}


def main():
    out = {"V0": V0, "S_ref": S_REF, "b_ref": B_REF, "note": __doc__, "symmetric": [], "asymmetric": []}
    for sy in (0.0, 0.1, 0.2):
        for sz in (0.0, 0.05, 0.10, 0.15):
            a = sy * U["sym_span"] + sz * U["sym_dihedral"]
            d = {"sym_span_m": sy, "sym_dihedral_m": sz, **derivs(a)}
            out["symmetric"].append(d)
            print(f"sy={sy:.2f} sz={sz:.2f}  Lb={d['L_beta']:7.2f} Nb={d['N_beta']:6.3f} Yb={d['Y_beta']:7.2f} "
                  f"Clb_NED={d['Cl_beta_NED']:+.3f} Cnb_NED={d['Cn_beta_NED']:+.4f} N_ds={d['N_ds']:+.3f} "
                  f"L_dy={d['L_dy']:.2f} arch={d['arch_height_m']:.3f}", flush=True)
    for key, amps in (("diff_span", (0.0, 0.05, 0.10, 0.15)), ("diff_sweep", (0.15, 0.30))):
        for d_ in amps:
            a = d_ * U[key]
            d = {"input": key, "amplitude_m": d_, **one_sided_Nbeta(a)}
            out["asymmetric"].append(d)
            print(f"{key:10s} {d_:.2f}  N_beta(beta<0)={d['N_beta_neg']:+.3f}  N_beta(beta>0)={d['N_beta_pos']:+.3f}",
                  flush=True)
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "lateral_derivatives.json"))


if __name__ == "__main__":
    main()
