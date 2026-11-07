#!/usr/bin/env python3
"""Airframe v3 (revision R1): physically derived rigid-body parameters.

Replaces the lumped inertia/damping of the v1/v2 airframe (Ixx = Iyy = 0.25, Izz = 0.12 kg m^2;
d_roll = d_pitch = 2.8, d_yaw = 0.25 N m s/rad) with values derived from the platform geometry:

  1. Mass properties from a component build-up (0.50 kg; 2.0 m span; 0.15 m chord; fuselage pod
     x = -0.34 ... +0.24 m as in the platform renders).
  2. Rotational damping. Roll and yaw damping of the wing are resolved by the strip model (rate-induced
     section velocities), so no lumped term is added. Pitch: the strip model samples the section
     velocity on the reference line only, which misses the section pitch damping; thin-airfoil
     theory for a section pitching about its quarter chord gives Cm_c/4 = -(pi/8) q c / V, i.e.
     d_pitch = (pi/16) rho V c^3 b.
  3. Static margin: CG placed 10 % of the chord ahead of the neutral point (typical tailless value).
  4. Trim: the nose-down moment of the stable CG is balanced by a reflexed-section zero-lift
     pitching moment cm0 (computed here for trim at V0, zero morphing input).
Output: revision_runs/analysis/airframe_v3.json
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402
from common import RUNS_DIR, dump_json  # noqa: E402
import numpy as np  # noqa: E402

from morphing_glider.config import L_FIXED, WING_CHORD  # noqa: E402
from morphing_glider.physics.domain_randomizer import NOMINAL_PHYS  # noqa: E402

SM_TARGET = 0.10
B = 2 * L_FIXED; C = WING_CHORD

# ---- 1. component build-up: (name, mass kg, x-extent [x0, x1] m, y-extent [y0, y1] m), uniform density ----
COMPONENTS = [
    ("wing structure (spars, ribs, covering), both halves", 0.16, (-C / 2, C / 2), (-L_FIXED, L_FIXED)),
    ("fuselage shell", 0.06, (-0.34, 0.24), (-0.03, 0.03)),
    ("battery, avionics, morphing actuators, payload", 0.28, (-0.15, 0.15), (-0.03, 0.03)),
]


def mass_properties():
    m_tot = sum(c[1] for c in COMPONENTS)
    x_c = sum(m * 0.5 * (x[0] + x[1]) for _, m, x, _ in COMPONENTS) / m_tot
    Ixx = Iyy = Izz = 0.0
    rows = []
    for name, m, (x0, x1), (y0, y1) in COMPONENTS:
        lx, ly = x1 - x0, y1 - y0; dx = 0.5 * (x0 + x1) - x_c
        ixx = m * ly ** 2 / 12; iyy = m * lx ** 2 / 12 + m * dx ** 2; izz = ixx + iyy
        Ixx += ixx; Iyy += iyy; Izz += izz
        rows.append({"component": name, "mass_kg": m, "x_range_m": [x0, x1], "y_range_m": [y0, y1],
                     "Ixx": ixx, "Iyy": iyy, "Izz": izz})
    return m_tot, {"Ixx": Ixx, "Iyy": Iyy, "Izz": Izz}, rows


def nominal_spars():
    from morphing_glider.training.infrastructure import make_env
    env = make_env(seed=0, domain_rand_scale=0.0, max_steps=10, for_eval=True, twist_enabled=True,
                   include_omega_cross=True)
    env.reset(seed=0); base = env.unwrapped
    return base.spar_R, base.spar_L, base.aero


def loads(spars, aero, phys, alpha, V):
    v = np.array([V * math.cos(alpha), 0.0, -V * math.sin(alpha)])
    F = np.zeros(3); M = np.zeros(3)
    for sp in spars:
        f, m, _ = aero.calculate_forces(sp, v_rel_body=v, omega_body=np.zeros(3), phys=phys)
        F += f; M += m
    return F, M


def main():
    m, I, rows = mass_properties()
    P = dict(NOMINAL_PHYS); P.update(x_cg=0.0, cm0=0.0)
    V0, rho, W = P["V0"], P["rho"], P["mass"] * P["g"]
    assert abs(m - P["mass"]) < 1e-9
    spR, spL, aero = nominal_spars(); spars = (spR, spL)
    # trim angle of attack (lift = weight, wing reference frame)
    a_lo, a_hi = -0.1, 0.3
    for _ in range(60):
        a = 0.5 * (a_lo + a_hi); Fz = loads(spars, aero, P, a, V0)[0][2]
        a_lo, a_hi = (a, a_hi) if Fz < W else (a_lo, a)
    a_trim = 0.5 * (a_lo + a_hi)
    da = 1e-3
    def My_of(a, ph): return loads(spars, aero, ph, a, V0)[1][1]
    L_alpha = (loads(spars, aero, P, a_trim + da, V0)[0][2] - loads(spars, aero, P, a_trim - da, V0)[0][2]) / (2 * da)
    K0 = (My_of(a_trim + da, P) - My_of(a_trim - da, P)) / (2 * da)      # +My = nose-down: K > 0 is stable
    x_np = -K0 / L_alpha                         # neutral point relative to the reference line (+ = ahead)
    x_cg = x_np + SM_TARGET * C
    P["x_cg"] = x_cg
    K = (My_of(a_trim + da, P) - My_of(a_trim - da, P)) / (2 * da)
    q0 = 0.5 * rho * V0 ** 2; S = B * C
    cm0 = My_of(a_trim, P) / (q0 * S * C)          # reflex needed to cancel the nose-down trim moment
    P["cm0"] = cm0
    My_trim = My_of(a_trim, P)
    d_pitch = math.pi / 16 * rho * V0 * C ** 3 * B
    # resolved aerodynamic rate damping at trim (finite differences on body rates)
    v_tr = np.array([V0 * math.cos(a_trim), 0.0, -V0 * math.sin(a_trim)])
    res = {}
    for i, ax in enumerate("pqr"):
        w = np.zeros(3); w[i] = 0.05
        Mp = sum(aero.calculate_forces(sp, v_rel_body=v_tr, omega_body=w, phys=P)[1] for sp in spars)
        Mm = sum(aero.calculate_forces(sp, v_rel_body=v_tr, omega_body=-w, phys=P)[1] for sp in spars)
        res[ax] = float(-(Mp[i] - Mm[i]) / 0.1)
    out = {
        "mass_budget": rows, "mass_kg": m, "inertia_kgm2": I,
        "trim": {"alpha_trim_rad": a_trim, "V0": V0, "L_alpha_N_per_rad": L_alpha,
                 "dMy_dalpha_ref_line": K0, "x_np_m": x_np, "x_cg_m": x_cg, "static_margin": SM_TARGET,
                 "dMy_dalpha_v3": K, "cm0_reflex": cm0, "My_trim_residual": My_trim},
        "damping": {"aero_resolved_N_m_s_per_rad": res, "d_pitch_section_thin_airfoil": d_pitch,
                    "d_roll": 0.0, "d_yaw": 0.0},
        "short_period_estimate": {"omega_rad_s": math.sqrt(max(K, 0) / I["Iyy"]),
                                  "roll_time_const_s": I["Ixx"] / max(res["p"], 1e-9),
                                  "yaw_time_const_s": I["Izz"] / max(res["r"], 1e-9)},
        "v2_lumped": {"Ixx": 0.25, "Iyy": 0.25, "Izz": 0.12, "d_roll": 2.8, "d_pitch": 2.8, "d_yaw": 0.25},
    }
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "airframe_v3.json"))
    import json; print(json.dumps({k: out[k] for k in ("inertia_kgm2", "trim", "damping", "short_period_estimate")}, indent=1))


if __name__ == "__main__":
    main()
