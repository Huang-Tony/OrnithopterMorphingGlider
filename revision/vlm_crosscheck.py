#!/usr/bin/env python3
"""Independent 3-D cross-check of the strip model with a vortex-lattice method (AeroSandbox VLM), revision R1.

Runs in a SEPARATE environment with AeroSandbox (not the simulator venv):
    <vlm_venv>/bin/python revision/vlm_crosscheck.py
Reads analysis/vlm_geometry.json (vlm_export.py: solved spar shapes + strip-model coefficients) and builds, for
every shape, the same wing: the solved spar is the quarter-chord line, chord 0.15 m, thin symmetric section with an
incidence equal to the strip model's zero-lift offset alpha0, reference point at the centre of mass, reference
area 0.30 m^2, span 2.0 m. The VLM is inviscid: its directional stability contains no profile-drag contribution.
A flat wing with +10 deg dihedral is run first as a sign check (conventional axes: Cl_beta < 0 is stabilizing).
Output: revision_runs/analysis/vlm_crosscheck.json
"""

import json
import math
import os
import sys

import aerosandbox as asb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(os.path.dirname(HERE), "revision_runs")
G = json.load(open(os.path.join(RUNS, "analysis", "vlm_geometry.json")))
S, B, C, V0 = G["S"], G["b"], G["c"], G["V0"]
INC = math.degrees(G["alpha0_rad"]); XCG = G["x_cg"]
AF = asb.Airfoil("naca0006")


def wing_from_points(pts):
    """pts: simulator body axes (x fwd, y span, z up) of the quarter-chord line, root to tip."""
    xs = []
    for p in pts:
        x_qc = -float(p[0])                       # AeroSandbox geometry axes: x aft
        xs.append(asb.WingXSec(xyz_le=[x_qc - 0.25 * C, abs(float(p[1])), float(p[2])], chord=C, twist=INC, airfoil=AF))
    return asb.Wing(name="wing", xsecs=xs, symmetric=True)


def run(wing):
    plane = asb.Airplane(name="glider", xyz_ref=[-XCG, 0.0, 0.0], wings=[wing], s_ref=S, c_ref=C, b_ref=B)
    op = asb.OperatingPoint(velocity=V0, alpha=0.0, beta=0.0)
    vlm = asb.VortexLatticeMethod(airplane=plane, op_point=op, spanwise_resolution=2, chordwise_resolution=8)
    d = vlm.run_with_stability_derivatives(alpha=True, beta=True, p=True, q=False, r=False)
    return {k: float(np.asarray(d[k]).ravel()[0]) for k in ("CL", "CLa", "CYb", "Clb", "Cnb", "Clp") if k in d}


def main():
    out = {"doc": __doc__, "sign_check": {}, "shapes": []}
    # sign check: flat wing, +10 deg dihedral, same chord/span
    y = np.linspace(0, 1.0, 9); pts = np.stack([np.zeros_like(y), y, y * math.tan(math.radians(10))], axis=1)
    sc = run(wing_from_points(pts)); out["sign_check"] = sc
    print(f"sign check (flat wing, +10 deg dihedral): Clb={sc['Clb']:+.4f} (expect < 0), CLa={sc['CLa']:.2f}", flush=True)
    for sh in G["shapes"]:
        r = run(wing_from_points(sh["spar_points"]))
        row = {"sym_span_m": sh["sym_span_m"], "sym_dihedral_m": sh["sym_dihedral_m"], "vlm": r,
               "strip": {k: sh[k] for k in ("CL", "CL_alpha", "Cl_beta", "Cn_beta", "CY_beta", "Cl_p")}}
        out["shapes"].append(row)
        print(f"sy={sh['sym_span_m']:.2f} sz={sh['sym_dihedral_m']:.2f} | Clb strip {sh['Cl_beta']:+.3f} vlm {r['Clb']:+.3f} | "
              f"CLa {sh['CL_alpha']:.2f}/{r['CLa']:.2f} | CYb {sh['CY_beta']:+.3f}/{r['CYb']:+.3f} | "
              f"Clp {sh['Cl_p']:+.3f}/{r['Clp']:+.3f} | Cnb {sh['Cn_beta']:+.4f}/{r['Cnb']:+.4f} | CL {sh['CL']:.3f}/{r['CL']:.3f}",
              flush=True)
    json.dump(out, open(os.path.join(RUNS, "analysis", "vlm_crosscheck.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
