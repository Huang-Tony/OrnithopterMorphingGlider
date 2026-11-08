#!/usr/bin/env python3
"""Model verification for the revised manuscript:
1. time-integration error of the 4 x 10 ms substep scheme and of a single 40 ms step against a 0.5 ms reference,
   under bounded multi-channel morphing inputs (open loop, no gusts or noise);
2. correlation of the Bezier control-polygon energy with Euler-Bernoulli bending energy (package validator);
3. figure fig_authority (yaw-moment map of single-wing deflection, differential-sweep saturation).
Writes revision_runs/analysis/model_verification.json and paper_figures/fig_authority.{pdf,png}.
"""

import math
import os

import numpy as np

import common  # noqa: F401
from common import RUNS_DIR, dump_json
from morphing_glider.config import L_FIXED
from morphing_glider.environment.env import MorphingGliderEnv6DOF
from morphing_glider.physics.aero_proxy import AeroProxy3D
from morphing_glider.physics.bezier_spar import RealTimeBezierSpar
from morphing_glider.physics.domain_randomizer import NOMINAL_PHYS
import figstyle
import matplotlib.pyplot as plt


def rollout(action_fn, n_sub, T=100):
    env = MorphingGliderEnv6DOF(max_steps=T, num_aero_panels=10, domain_rand_scale=0.0, seed=3,
                                roll_pitch_limit_deg=89.0, sensor_noise_scale=0.0, physics_substeps=n_sub)
    env.reset(seed=3); env.phys.update(gust_sigma_x=0.0, gust_sigma_y=0.0, gust_sigma_z=0.0)
    for s in (env.spar_R, env.spar_L): s.iterations = 12
    out = []
    for t in range(T):
        _, _, term, _, info = env.step(np.asarray(action_fn(t), dtype=np.float32))
        out.append([info["roll"], info["pitch"], info["yaw_rate"]])
        if term: break
    return np.array(out)


def integration_check():
    w = 2 * math.pi * 1.0
    cases = {
        "sinusoidal_1Hz": lambda t: np.array([0.2 * math.sin(w * t * 0.04), 0, 0.05 * math.sin(w * t * 0.04),
                                              -0.2 * math.sin(w * t * 0.04), 0, -0.05 * math.sin(w * t * 0.04)]),
        "multisine": lambda t: np.array([0.15 * math.sin(2.1 * t * 0.04 * 6.28) + 0.1 * math.sin(0.7 * t * 0.04 * 6.28), 0.03 * math.sin(1.3 * t * 0.04 * 6.28),
                                         0.04 * math.sin(1.7 * t * 0.04 * 6.28), -0.12 * math.sin(2.1 * t * 0.04 * 6.28), -0.03 * math.sin(1.3 * t * 0.04 * 6.28),
                                         -0.02 * math.sin(0.9 * t * 0.04 * 6.28)]),
    }
    res = {}
    A = 0.35   # input scale (airframe v3 rolls ~1.7x faster than v2; keeps the open-loop runs below 89 deg longer)
    cases = {k: (lambda t, f=f: A * f(t)) for k, f in cases.items()}
    for name, fn in cases.items():
        ref = rollout(fn, 80); sub = rollout(fn, 4); one = rollout(fn, 1)
        n = min(len(ref), len(sub)); n1 = min(len(ref), len(one))       # each scheme over its own window
        amp = np.sqrt(np.mean(ref[:n, 2] ** 2)); amp1 = np.sqrt(np.mean(ref[:n1, 2] ** 2))
        res[name] = {"steps": int(n), "steps_sub1": int(n1), "yaw_rate_rms_ref": float(amp),
                     "yaw_err_pct_sub4": float(100 * np.sqrt(np.mean((sub[:n, 2] - ref[:n, 2]) ** 2)) / amp),
                     "yaw_err_pct_sub1": float(100 * np.sqrt(np.mean((one[:n1, 2] - ref[:n1, 2]) ** 2)) / amp1),
                     "roll_err_deg_sub4": float(np.degrees(np.max(np.abs(sub[:n, 0] - ref[:n, 0])))),
                     "roll_err_deg_sub1": float(np.degrees(np.max(np.abs(one[:n1, 0] - ref[:n1, 0]))))}
        print(name, {k: round(v, 3) for k, v in res[name].items()})
    return res


def closed_loop_check(T=200):
    """Closed-loop verification over a full 8 s episode with yaw-reference steps: the heuristic controller flies
    the nominal vehicle (no gusts or noise); identical controller and reference, different integration steps."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from classical import make_classical

    def run(n_sub):
        env = MorphingGliderEnv6DOF(max_steps=T, num_aero_panels=10, domain_rand_scale=0.0, seed=5,
                                    roll_pitch_limit_deg=89.0, sensor_noise_scale=0.0, physics_substeps=n_sub)
        obs, _ = env.reset(seed=5); env.phys.update(gust_sigma_x=0.0, gust_sigma_y=0.0, gust_sigma_z=0.0)
        for s_ in (env.spar_R, env.spar_L): s_.iterations = 12
        c = make_classical("Heuristic"); out = []
        for t in range(T):
            a, _ = c.predict(obs)
            obs, _, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
            out.append([info["roll"], info["pitch"], info["yaw_rate"]])
            if term or trunc: break
        return np.array(out)
    ref, sub = run(80), run(4); n = min(len(ref), len(sub))
    amp = np.sqrt(np.mean(ref[:n, 2] ** 2))
    res = {"steps": int(n), "yaw_rate_rms_ref": float(amp),
           "yaw_err_pct_sub4": float(100 * np.sqrt(np.mean((sub[:n, 2] - ref[:n, 2]) ** 2)) / amp),
           "roll_err_deg_sub4": float(np.degrees(np.max(np.abs(sub[:n, 0] - ref[:n, 0])))),
           "pitch_err_deg_sub4": float(np.degrees(np.max(np.abs(sub[:n, 1] - ref[:n, 1]))))}
    print("closed loop", {k: round(v, 3) for k, v in res.items()})
    return res


def spar_energy_check():
    try:
        from morphing_glider.physics.validators import validate_spar_proxy
        r = validate_spar_proxy(n_deflections=8)
        return {"correlation": float(r.get("correlation", float("nan")))}
    except Exception as e:
        return {"error": repr(e)}


def solve(p3):
    s = 1.0 if p3[1] >= 0 else -1.0
    sp = RealTimeBezierSpar([0, 0, 0], p3, [0, s * 0.33, 0], [0, s * 0.66, 0]); sp.iterations = 12
    sp.solve_to_convergence(max_total_iters=144, chunk_iters=12, tol_len=1e-4); return sp


def authority_figure(yaw_damping):
    phys = dict(NOMINAL_PHYS); aero = AeroProxy3D(num_panels=10); v = np.array([phys["V0"], 0, 0])
    L0, R0 = solve([0, -L_FIXED, 0]), solve([0, L_FIXED, 0])
    M0 = aero.calculate_forces(R0, v_rel_body=v, omega_body=np.zeros(3), phys=phys)[1] + aero.calculate_forces(L0, v_rel_body=v, omega_body=np.zeros(3), phys=phys)[1]
    xs = np.linspace(0, 0.45, 19); zs = np.linspace(0, 0.15, 13); G = np.zeros((zs.size, xs.size))
    for i, z in enumerate(zs):
        for j, x in enumerate(xs):
            R = solve([x, L_FIXED, z])
            G[i, j] = (aero.calculate_forces(R, v_rel_body=v, omega_body=np.zeros(3), phys=phys)[1] + aero.calculate_forces(L0, v_rel_body=v, omega_body=np.zeros(3), phys=phys)[1])[2] - M0[2]
    ds = np.linspace(0, 0.5, 21); mz = []
    for d in ds:
        R = solve([d, L_FIXED, 0]); L = solve([-d, -L_FIXED, 0])
        mz.append((aero.calculate_forces(R, v_rel_body=v, omega_body=np.zeros(3), phys=phys)[1] + aero.calculate_forces(L, v_rel_body=v, omega_body=np.zeros(3), phys=phys)[1])[2] - M0[2])
    mz = np.array(mz)
    figstyle.apply()
    fig, axs = plt.subplots(1, 2, figsize=(figstyle.DOUBLE, 2.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    im = ax.pcolormesh(xs, zs, G, shading="gouraud", cmap="Blues", vmin=0)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.06); cb.set_label(r"$\Delta M_z$ (N m)"); cb.outline.set_linewidth(0.4)
    cs = ax.contour(xs, zs, G, levels=[0.01, 0.02, 0.03, 0.04, 0.05], colors="white", linewidths=0.4)
    ax.clabel(cs, fmt="%.2f", fontsize=6)
    ax.set_xlabel(r"tip sweep $\Delta x$ (m)"); ax.set_ylabel(r"tip dihedral $\Delta z$ (m)"); figstyle.panel_label(ax, "a", x=-0.2)
    ax = axs[1]
    ax.plot(ds, mz, color=figstyle.SLOT["blue"], lw=1.0)
    ax.axhline(mz.max(), color=figstyle.MUTED, lw=0.6, ls="--")
    ax.set_xlabel(r"differential sweep amplitude $\delta$ (m)"); ax.set_ylabel(r"$\Delta M_z$ (N m)")
    # airframe v3: no steady-yaw-rate axis (with weak yaw damping the response is governed by sideslip and bank)
    ax.text(0.02, mz.max() * 1.03, f"saturation {mz.max():.3f} N m", fontsize=6.3, color=figstyle.INK2, va="bottom")
    ax.set_ylim(0, mz.max() * 1.18); figstyle.panel_label(ax, "b", x=-0.22)
    fig.tight_layout(w_pad=2.5); figstyle.save(fig, os.path.join(RUNS_DIR, "paper_figures", "fig_authority"))
    return {"single_wing_max": float(G.max()), "diff_sweep_max": float(mz.max())}


def main():
    import json
    pi = json.load(open(os.path.join(RUNS_DIR, "analysis", "plant_identification.json")))
    out = {"integration": integration_check(), "closed_loop": closed_loop_check(), "spar_energy": spar_energy_check(),
           "authority": authority_figure(pi["yaw_model"]["total_yaw_damping"])}
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "model_verification.json"))
    print("spar energy:", out["spar_energy"], "authority:", out["authority"])


if __name__ == "__main__":
    main()
