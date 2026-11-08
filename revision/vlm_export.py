#!/usr/bin/env python3
"""Export the solved spar shapes and the strip-model derivatives for the independent vortex-lattice cross-check
(revision R1; revision/vlm_crosscheck.py runs in a separate environment with AeroSandbox).

Shapes: the symmetric pre-shapes of Table tab:lateral. For each: the solved right-wing spar sampled at 25 points
(simulator body axes: x forward, y toward the +y wing, z up), and the strip-model coefficients at V0, body alpha = 0,
in conventional aircraft (NED) axes: CL, CL_alpha, Cl_beta, Cn_beta, CY_beta, Cl_p (p normalized by b/2V).
Output: revision_runs/analysis/vlm_geometry.json
"""

import math
import os

import common  # noqa: F401
from common import RUNS_DIR, dump_json
import numpy as np
import identify_plant as ip

S, B, C = 0.30, 2.0, 0.15
V0 = ip.V0; QS = 0.5 * ip.PHYS["rho"] * V0 ** 2 * S
SHAPES = [(0.0, 0.0), (0.0, 0.05), (0.0, 0.10), (0.0, 0.15), (0.1, 0.0), (0.1, 0.10), (0.2, 0.05)]


def loads(a, v=None, w=np.zeros(3)):
    F, M, L, D = ip.loads(a, v_body=v, omega=w)
    return F, M


def main():
    out = {"S": S, "b": B, "c": C, "V0": V0, "alpha0_rad": float(ip.PHYS["alpha0"]), "x_cg": float(ip.PHYS["x_cg"]), "shapes": []}
    for sy, sz in SHAPES:
        a = sy * np.asarray(ip.VIRTUAL["sym_span"], float) + sz * np.asarray(ip.VIRTUAL["sym_dihedral"], float)
        sp = ip.spar([a[0], ip.L_FIXED + a[1], a[2]])
        pts = sp.evaluate(np.linspace(0.0, 1.0, 25)[:, None])
        F0, M0 = loads(a, v=V0 * np.array([1.0, 0, 0]))
        da = math.radians(1.0)
        Fp, _ = loads(a, v=V0 * np.array([math.cos(da), 0, -math.sin(da)])); Fm, _ = loads(a, v=V0 * np.array([math.cos(da), 0, math.sin(da)]))
        db = math.radians(2.0)
        Fbp, Mbp = loads(a, v=V0 * np.array([math.cos(db), math.sin(db), 0])); Fbm, Mbm = loads(a, v=V0 * np.array([math.cos(db), -math.sin(db), 0]))
        wp = np.array([0.05, 0, 0]); _, Mpp = loads(a, w=wp); _, Mpm = loads(a, w=-wp)
        # simulator body axes (x fwd, y to +y wing, z up) -> NED: L_ned = L, N_ned = -N, Y_ned = -Y, beta_ned = -beta
        Lb, Nb, Yb = (Mbp[0] - Mbm[0]) / (2 * db), (Mbp[2] - Mbm[2]) / (2 * db), (Fbp[1] - Fbm[1]) / (2 * db)
        out["shapes"].append({
            "sym_span_m": sy, "sym_dihedral_m": sz, "spar_points": pts.tolist(),
            "CL": float(F0[2] / QS), "CL_alpha": float((Fp[2] - Fm[2]) / (2 * da) / QS),
            "Cl_beta": float(-Lb / (QS * B)), "Cn_beta": float(Nb / (QS * B)), "CY_beta": float(Yb / QS),
            "Cl_p": float(((Mpp[0] - Mpm[0]) / 0.1) * 2 * V0 / (QS * B * B))})
        r = out["shapes"][-1]
        print(f"sy={sy:.2f} sz={sz:.2f} CL={r['CL']:.3f} CLa={r['CL_alpha']:.2f} Clb={r['Cl_beta']:+.3f} Cnb={r['Cn_beta']:+.4f} "
              f"CYb={r['CY_beta']:+.3f} Clp={r['Cl_p']:+.3f}", flush=True)
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "vlm_geometry.json"))


if __name__ == "__main__":
    main()
