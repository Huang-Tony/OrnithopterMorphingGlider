#!/usr/bin/env python3
"""Solve the spar shapes (simulator's energy solver) and aerodynamic moments of the wing configurations
shown in the platform figure; writes revision_runs/analysis/platform_configs.json for the 3D renderer."""

import os

import numpy as np

import common  # noqa: F401
from common import RUNS_DIR, dump_json
from morphing_glider.config import L_FIXED, WING_CHORD
from morphing_glider.physics.aero_proxy import AeroProxy3D
from morphing_glider.physics.bezier_spar import RealTimeBezierSpar
from morphing_glider.physics.domain_randomizer import NOMINAL_PHYS

CONFIGS = {   # tip offsets [dx_R, dy_R, dz_R, dx_L, dy_L, dz_L]  (R = +y wing)
    "neutral":        [0, 0, 0, 0, 0, 0],
    "diff_sweep":     [0.30, 0, 0, -0.30, 0, 0],      # yaw effector
    "diff_dihedral":  [0, 0, 0.12, 0, 0, -0.12],     # roll effector
    "hero":           [0.12, 0, 0.06, -0.20, 0, -0.03],
}


def solve(p3):
    s = 1.0 if p3[1] >= 0 else -1.0
    sp = RealTimeBezierSpar([0, 0, 0], p3, [0, s * 0.33, 0], [0, s * 0.66, 0]); sp.iterations = 12
    sp.solve_to_convergence(max_total_iters=144, chunk_iters=12, tol_len=1e-4)
    return sp


def main():
    phys = dict(NOMINAL_PHYS); aero = AeroProxy3D(num_panels=10); v = np.array([phys["V0"], 0.0, 0.0])
    out = {"chord": WING_CHORD, "semi_span": L_FIXED, "configs": {}}
    M0 = None
    for name, a in CONFIGS.items():
        a = np.asarray(a, float)
        R = solve([a[0], L_FIXED + a[1], a[2]]); L = solve([a[3], -L_FIXED + a[4], a[5]])
        FR, MR, _ = aero.calculate_forces(R, v_rel_body=v, omega_body=np.zeros(3), phys=phys)
        FL, ML, _ = aero.calculate_forces(L, v_rel_body=v, omega_body=np.zeros(3), phys=phys)
        M = MR + ML
        if M0 is None: M0 = M.copy()
        out["configs"][name] = {"action": a.tolist(),
                                "R": [R.p0.tolist(), R.p1.tolist(), R.p2.tolist(), R.p3.tolist()],
                                "L": [L.p0.tolist(), L.p1.tolist(), L.p2.tolist(), L.p3.tolist()],
                                "dM": (M - M0).tolist(), "F": (FR + FL).tolist()}
        print(f"{name:<14} dM = {np.round(M - M0, 4)}")
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "platform_configs.json"))


if __name__ == "__main__":
    main()
