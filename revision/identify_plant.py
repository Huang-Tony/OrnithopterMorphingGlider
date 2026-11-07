#!/usr/bin/env python3
"""Static flight-mechanics analysis of the (corrected) glider model at the initial trim condition.

Computes, by central finite differences of the aerodynamic proxy:
  * control effectiveness dM/d(delta) [N*m/m] and dF/d(delta) [N/m] of five morphing 'virtual inputs':
      differential sweep, symmetric sweep, differential dihedral, symmetric dihedral, differential span;
  * rate-damping derivatives dM/d(p,q,r) (aerodynamic part; the lumped linear damping D is added separately);
  * sideslip and angle-of-attack derivatives dM/d(beta), dM/d(alpha), dF/d(alpha);
  * trim quantities (lift, drag, L/D, pitching moment).
Writes revision_runs/analysis/plant_identification.json (used for controller design and in the paper).
"""

import math
import os

import common  # noqa: F401
from common import RUNS_DIR, dump_json
import numpy as np
from morphing_glider.config import L_FIXED
from morphing_glider.physics.aero_proxy import AeroProxy3D
from morphing_glider.physics.bezier_spar import RealTimeBezierSpar
from morphing_glider.physics.domain_randomizer import NOMINAL_PHYS

PHYS = dict(NOMINAL_PHYS)
V0 = float(PHYS["V0"])
AERO = AeroProxy3D(num_panels=10, include_omega_cross=True)

# virtual inputs -> tip offsets [dx_R, dy_R, dz_R, dx_L, dy_L, dz_L]  (R = +y wing, L = -y wing)
VIRTUAL = {
    "diff_sweep":    np.array([+1, 0, 0, -1, 0, 0], float),
    "sym_sweep":     np.array([+1, 0, 0, +1, 0, 0], float),
    "diff_dihedral": np.array([0, 0, +1, 0, 0, -1], float),
    "sym_dihedral":  np.array([0, 0, +1, 0, 0, +1], float),
    "diff_span":     np.array([0, +1, 0, 0, +1, 0], float),   # y is mirrored: R tip out, L tip in
    "sym_span":      np.array([0, +1, 0, 0, -1, 0], float),   # both tips outward
}
STEP = {"diff_sweep": 0.05, "sym_sweep": 0.05, "diff_dihedral": 0.02, "sym_dihedral": 0.02, "diff_span": 0.02,
        "sym_span": 0.02}


def spar(p3):
    s = 1.0 if p3[1] >= 0 else -1.0
    sp = RealTimeBezierSpar([0, 0, 0], p3, [0, s * 0.33 * L_FIXED, 0], [0, s * 0.66 * L_FIXED, 0])
    sp.iterations = 12; sp.solve_to_convergence(max_total_iters=144, chunk_iters=12, tol_len=1e-4)
    return sp


def loads(a, v_body=None, omega=np.zeros(3)):
    a = np.asarray(a, float)
    sR = spar([a[0], L_FIXED + a[1], a[2]]); sL = spar([a[3], -L_FIXED + a[4], a[5]])
    v = np.array([V0, 0.0, 0.0]) if v_body is None else v_body
    FR, MR, dR = AERO.calculate_forces(sR, v_rel_body=v, omega_body=omega, phys=PHYS)
    FL, ML, dL = AERO.calculate_forces(sL, v_rel_body=v, omega_body=omega, phys=PHYS)
    return FR + FL, MR + ML, dR["total_lift_force"] + dL["total_lift_force"], dR["total_drag_force"] + dL["total_drag_force"]


def main():
    out = {"V0": V0, "phys": PHYS}
    F0, M0, L0, D0 = loads(np.zeros(6))
    W = PHYS["mass"] * PHYS["g"]
    out["trim"] = {"F_body": F0, "M_body": M0, "lift_N": L0, "drag_N": D0, "L_over_D": L0 / max(D0, 1e-9),
                   "weight_N": W, "lift_over_weight": L0 / W}
    eff = {}
    for k, u in VIRTUAL.items():
        h = STEP[k]
        Fp, Mp, _, Dp = loads(+h * u); Fm, Mm, _, Dm = loads(-h * u)
        eff[k] = {"dM_ddelta": (Mp - Mm) / (2 * h), "dF_ddelta": (Fp - Fm) / (2 * h),
                  "dDrag_ddelta": (Dp - Dm) / (2 * h), "step_m": h,
                  "M_at_plus_0p2": loads(0.2 * u)[1] - M0 if k.endswith("sweep") else loads(0.1 * u)[1] - M0}
    out["control_effectiveness"] = eff
    # rate damping (aerodynamic part)
    rd = {}
    for i, name in enumerate(["p", "q", "r"]):
        w = np.zeros(3); w[i] = 0.05
        _, Mp, _, _ = loads(np.zeros(6), omega=w); _, Mm, _, _ = loads(np.zeros(6), omega=-w)
        rd[name] = (Mp - Mm) / 0.1
    out["aero_rate_derivatives_dM_domega"] = rd
    out["lumped_linear_damping"] = {"d_roll": PHYS["d_roll"], "d_pitch": PHYS["d_pitch"], "d_yaw": PHYS["d_yaw"]}
    # sideslip / angle of attack
    b = math.radians(2.0)
    vp = V0 * np.array([math.cos(b), math.sin(b), 0.0]); vm = V0 * np.array([math.cos(b), -math.sin(b), 0.0])
    Fbp, Mbp, _, _ = loads(np.zeros(6), v_body=vp); Fbm, Mbm, _, _ = loads(np.zeros(6), v_body=vm)
    out["dM_dbeta_per_rad"] = (Mbp - Mbm) / (2 * b); out["dF_dbeta_per_rad"] = (Fbp - Fbm) / (2 * b)
    a = math.radians(1.0)
    vp = V0 * np.array([math.cos(a), 0.0, -math.sin(a)]); vm = V0 * np.array([math.cos(a), 0.0, math.sin(a)])
    Fap, Map, Lap, _ = loads(np.zeros(6), v_body=vp); Fam, Mam, Lam, _ = loads(np.zeros(6), v_body=vm)
    out["dM_dalpha_per_rad"] = (Map - Mam) / (2 * a); out["dLift_dalpha_per_rad"] = (Lap - Lam) / (2 * a)
    qS = 0.5 * PHYS["rho"] * V0 ** 2 * 0.3
    out["CL_alpha_whole_wing_per_rad"] = out["dLift_dalpha_per_rad"] / qS
    # first-order yaw model per unit differential sweep (for SISO baselines)
    D_r = PHYS["d_yaw"] - rd["r"][2]
    # large-amplitude (nonlinear) yaw authority of differential sweep
    out["diff_sweep_yaw_moment_curve"] = [{"delta_m": d, "Mz": float(loads(d * VIRTUAL["diff_sweep"])[1][2] - M0[2])}
                                          for d in (0.1, 0.2, 0.3, 0.4, 0.5)]
    out["yaw_model"] = {"Izz": PHYS["Izz"], "N_delta_diff_sweep": eff["diff_sweep"]["dM_ddelta"][2],
                        "total_yaw_damping": D_r, "time_constant_s": PHYS["Izz"] / D_r,
                        "steady_gain_rad_s_per_m": eff["diff_sweep"]["dM_ddelta"][2] / D_r}
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "plant_identification.json"))
    np.set_printoptions(precision=4, suppress=True)
    print(f"trim: lift={L0:.3f} N (W={W:.3f}), drag={D0:.3f} N, L/D={L0/D0:.2f}, M0={M0}")
    for k, e in eff.items():
        print(f"{k:<14} dM/dδ={np.round(e['dM_ddelta'],4)} N*m/m   dF/dδ={np.round(e['dF_ddelta'],3)} N/m")
    for k, v in rd.items(): print(f"aero dM/d{k} = {np.round(v,4)} N*m*s/rad")
    print("dM/dbeta =", np.round(out["dM_dbeta_per_rad"], 4), " dM/dalpha =", np.round(out["dM_dalpha_per_rad"], 4),
          " CL_alpha(wing) =", round(out["CL_alpha_whole_wing_per_rad"], 3))
    print("diff-sweep Mz curve:", [(c["delta_m"], round(c["Mz"], 4)) for c in out["diff_sweep_yaw_moment_curve"]])
    print("yaw model:", {k: round(float(v), 4) for k, v in out["yaw_model"].items()})


if __name__ == "__main__":
    main()
