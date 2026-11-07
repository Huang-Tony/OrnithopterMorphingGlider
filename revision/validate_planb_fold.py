#!/usr/bin/env python3
"""Plan B validation with the author's own hardware data (companion ornithopter study, fold-actuation test:
servo-driven compliant fold joint, right wing, Vicon at 100 Hz, ~/Desktop/ORNITHOPTER DATA, read-only).

1. Actuator dynamics: for every commanded step on the steep part of the static map, fit (a) a first-order
   lag with dead time, tau*dq/dt + q = u(t - d), and (b) a second-order model, to the normalised fold
   response; compare with the simulator's first-order actuator, tau = 0.07 s.
2. Spar-shape model: fit straight line / quadratic / cubic Bezier curves (fixed end points) to the 15
   leading-edge spar markers in every frame, and test whether the simulator's energy solver, given only
   the measured root and tip, predicts the interior shape.
Outputs: revision_runs/analysis/planb_fold_validation.json, paper figure fig_planb_fold.{pdf,png}.
"""

import csv
import json
import math
import os

import numpy as np
from scipy.optimize import least_squares

import common  # noqa: F401
from common import RUNS_DIR, dump_json
import figstyle
import matplotlib.pyplot as plt

SRC = os.path.expanduser("~/Desktop/ORNITHOPTER DATA/analysis/output/servo_range")
DT = 0.01


def read_csv(name):
    with open(os.path.join(SRC, name)) as f:
        r = csv.DictReader(f); return list(r)


def fopdt(params, t, u_fun):
    tau, d = params; q = np.zeros_like(t); a = math.exp(-DT / max(tau, 1e-4))
    for i in range(1, t.size):
        q[i] = a * q[i - 1] + (1 - a) * u_fun(t[i - 1] - d)
    return q


def sopdt(params, t, u_fun):
    wn, zeta, d = params; x = 0.0; v = 0.0; q = np.zeros_like(t); h = DT / 10
    for i in range(1, t.size):
        for _ in range(10):
            u = u_fun(t[i - 1] - d)
            acc = wn * wn * (u - x) - 2 * zeta * wn * v
            v += acc * h; x += v * h
        q[i] = x
    return q


def actuator_fits():
    resp = read_csv("servo_response.csv"); steps = read_csv("servo_steps.csv"); sched = read_csv("servo_schedule.csv")
    t_all = np.array([float(r["Time_s"]) for r in resp]); q_all = np.array([float(r["fold_q"]) for r in resp])
    sch = {(r["test"], r["step"]): r for r in sched}
    fits = []
    for s in steps:
        try:
            dq = float(s["dq"]); speed = float(s["speed_eff_deg_s"])
        except ValueError:
            continue
        if abs(dq) < 0.05 or s["kind"] != "up": continue
        sc = sch.get((s["test"], s["step"]))
        if sc is None: continue
        t0, t1 = float(sc["t0_capture"]), float(sc["t1_capture"])
        q0, q1 = float(s["q_before"]), float(s["q_plateau"])
        win = (t_all >= t0 - 0.10) & (t_all <= t1 + 0.60)
        t = t_all[win] - t0; qn = (q_all[win] - q0) / (q1 - q0)
        T = max(t1 - t0, 1e-3)
        u_fun = lambda tt, T=T: float(np.clip(tt / T, 0.0, 1.0))
        r1 = least_squares(lambda p: fopdt(p, t, u_fun) - qn, x0=[0.05, 0.01], bounds=([0.002, -0.05], [1.0, 0.2]))
        r2 = least_squares(lambda p: sopdt(p, t, u_fun) - qn, x0=[15.0, 0.6, 0.01], bounds=([2.0, 0.05, -0.05], [200.0, 3.0, 0.2]))
        sim07 = fopdt([0.07, 0.0], t, u_fun)
        fits.append({"test": int(s["test"]), "step": int(s["step"]), "speed_deg_s": speed, "ramp_s": T, "dq": dq,
                     "tau_s": float(r1.x[0]), "delay_s": float(r1.x[1]), "rmse_first_order": float(np.sqrt(np.mean(r1.fun ** 2))),
                     "wn_rad_s": float(r2.x[0]), "zeta": float(r2.x[1]), "delay2_s": float(r2.x[2]),
                     "rmse_second_order": float(np.sqrt(np.mean(r2.fun ** 2))),
                     "rmse_sim_tau007": float(np.sqrt(np.mean((sim07 - qn) ** 2))),
                     "overshoot": float(max(0.0, qn.max() - 1.0)),
                     "_t": t.tolist(), "_qn": qn.tolist(), "_fit1": (r1.fun + qn).tolist(), "_sim07": sim07.tolist()})
    return fits


def bezier(P0, P1, P2, P3, u):
    u = u[:, None]
    return (1 - u) ** 3 * P0 + 3 * (1 - u) ** 2 * u * P1 + 3 * (1 - u) * u ** 2 * P2 + u ** 3 * P3


def fit_curves(pts):
    """pts: (15,3) root->tip. Returns RMS (mm) of line / quadratic / cubic fits with fixed end points."""
    P0, P3 = pts[0], pts[-1]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)]); u = s / s[-1]
    line = P0 + u[:, None] * (P3 - P0)
    rms_line = float(np.sqrt(np.mean(np.sum((pts - line) ** 2, axis=1))))
    # quadratic: B = (1-u)^2 P0 + 2u(1-u) Q + u^2 P3  -> linear LS for Q
    b = 2 * u * (1 - u); rhs = pts - ((1 - u) ** 2)[:, None] * P0 - (u ** 2)[:, None] * P3
    Q = (b[:, None] * rhs).sum(0) / max((b ** 2).sum(), 1e-12)
    quad = ((1 - u) ** 2)[:, None] * P0 + b[:, None] * Q + (u ** 2)[:, None] * P3
    rms_quad = float(np.sqrt(np.mean(np.sum((pts - quad) ** 2, axis=1))))
    # cubic: linear LS for P1, P2, then 3 rounds of parameter correction (closest point)
    for _ in range(4):
        A = np.stack([3 * (1 - u) ** 2 * u, 3 * (1 - u) * u ** 2], axis=1)
        rhs = pts - ((1 - u) ** 3)[:, None] * P0 - (u ** 3)[:, None] * P3
        X, *_ = np.linalg.lstsq(A, rhs, rcond=None); P1, P2 = X[0], X[1]
        uu = np.linspace(0, 1, 400); C = bezier(P0, P1, P2, P3, uu)
        idx = np.argmin(np.linalg.norm(pts[:, None, :] - C[None], axis=2), axis=1); u_new = uu[idx]
        u_new[0], u_new[-1] = 0.0, 1.0; u = u_new
    C = bezier(P0, P1, P2, P3, np.linspace(0, 1, 400))
    d_cub = np.min(np.linalg.norm(pts[:, None, :] - C[None], axis=2), axis=1)
    rms_cub = float(np.sqrt(np.mean(d_cub ** 2)))
    return rms_line, rms_quad, rms_cub, (P1, P2), float(s[-1])


def solver_prediction(pts, w_bio):
    """Simulator energy solver given only measured root/tip (scaled to unit arc length)."""
    from morphing_glider.physics.bezier_spar import RealTimeBezierSpar
    P0, P3 = pts[0], pts[-1]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); L = float(seg.sum())
    p3 = (P3 - P0) / L
    sp = RealTimeBezierSpar([0, 0, 0], p3, 0.33 * p3, 0.66 * p3); sp.iterations = 12
    for _ in range(40):
        sp.solve_shape(iterations=12, w_bio=w_bio)
    C = P0 + L * bezier(np.zeros(3), sp.p1, sp.p2, p3, np.linspace(0, 1, 400))
    d = np.min(np.linalg.norm(pts[:, None, :] - C[None], axis=2), axis=1)
    return float(np.sqrt(np.mean(d ** 2))), C


def main():
    figstyle.apply(); out = {"source": SRC}
    fits = actuator_fits()
    taus = np.array([f["tau_s"] for f in fits]); delays = np.array([f["delay_s"] for f in fits])
    out["actuator"] = {"n_steps": len(fits), "tau_median_s": float(np.median(taus)), "tau_iqr_s": np.percentile(taus, [25, 75]).tolist(),
                       "delay_median_s": float(np.median(delays)),
                       "wn_median_rad_s": float(np.median([f["wn_rad_s"] for f in fits])),
                       "zeta_median": float(np.median([f["zeta"] for f in fits])),
                       "rmse_first_order_median": float(np.median([f["rmse_first_order"] for f in fits])),
                       "rmse_second_order_median": float(np.median([f["rmse_second_order"] for f in fits])),
                       "rmse_sim_tau007_median": float(np.median([f["rmse_sim_tau007"] for f in fits])),
                       "per_step": [{k: v for k, v in f.items() if not k.startswith("_")} for f in fits]}
    try:
        ring = read_csv("servo_ringing.csv")
        fn = [float(r["f_hz"]) for r in ring if r["signal"].startswith("mode 2")]
        z = [float(r["zeta"]) for r in ring if r["signal"].startswith("mode 2")]
        out["wing_mode"] = {"f_n_hz_median": float(np.median(fn)), "zeta_median": float(np.median(z)), "n": len(fn)}
    except Exception as e:
        out["wing_mode"] = {"error": repr(e)}
    try:
        relax = read_csv("servo_relaxation.csv"); T = [float(r["T_s"]) for r in relax]
        out["relaxation_time_constant_s"] = {"median": float(np.median(T)), "min": float(min(T)), "max": float(max(T))}
    except Exception as e:
        out["relaxation_time_constant_s"] = {"error": repr(e)}

    # ---- spar shape ----
    with open(os.path.join(SRC, "markers_wingframe_filled.csv")) as f:
        r = csv.reader(f); hdr = next(r); rows = list(r)
    names = [f"RA{i:02d}" for i in range(1, 16)]
    cols = [[hdr.index(f"{n}_{c}") for c in "XYZ"] for n in names]
    q_resp = {round(float(rr["Time_s"]), 2): float(rr["fold_q"]) for rr in read_csv("servo_response.csv")}
    recs = []
    for k in range(0, len(rows), 4):
        row = rows[k]
        try:
            pts = np.array([[float(row[c]) for c in cc] for cc in cols])
        except ValueError:
            continue
        if not np.all(np.isfinite(pts)): continue
        rl, rq, rc, _, L = fit_curves(pts)
        rs0, _ = solver_prediction(pts, 0.0)
        recs.append({"t": float(row[1]), "q": q_resp.get(round(float(row[1]), 2), np.nan), "L_mm": L,
                     "rms_line_mm": rl, "rms_quad_mm": rq, "rms_cubic_mm": rc, "rms_solver_mm": rs0})
    R = {k: np.array([x[k] for x in recs]) for k in recs[0]}
    out["spar_shape"] = {"n_frames": len(recs), "arc_length_mm_median": float(np.median(R["L_mm"])),
                         **{f"{k}_median": float(np.nanmedian(R[k])) for k in ("rms_line_mm", "rms_quad_mm", "rms_cubic_mm", "rms_solver_mm")},
                         **{f"{k}_p95": float(np.nanpercentile(R[k], 95)) for k in ("rms_line_mm", "rms_quad_mm", "rms_cubic_mm", "rms_solver_mm")},
                         "q_range": [float(np.nanmin(R["q"])), float(np.nanmax(R["q"]))]}
    # representative frames: rest and maximum fold
    iq = np.nanargmax(R["q"]); i0 = int(np.nanargmin(np.abs(R["q"])))
    rep = {}
    for tag, i in (("rest", i0), ("max_fold", iq)):
        row = rows[[k for k in range(0, len(rows), 4)][0]] if False else None
        t_i = recs[i]["t"]; k = int(round((t_i - float(rows[0][1])) / DT)); row = rows[min(max(k, 0), len(rows) - 1)]
        pts = np.array([[float(row[c]) for c in cc] for cc in cols])
        _, _, _, (P1, P2), _ = fit_curves(pts)
        _, Csol = solver_prediction(pts, 0.0); _, Cbio = solver_prediction(pts, 2.0)
        rep[tag] = {"pts": pts.tolist(), "cubic": bezier(pts[0], P1, P2, pts[-1], np.linspace(0, 1, 200)).tolist(),
                    "solver": Csol[::2].tolist(), "solver_bio": Cbio[::2].tolist(), "q": recs[i]["q"]}
    out["representative_frames"] = rep
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "planb_fold_validation.json"))

    # ---- figure ----
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    ax = axs[0]; ex = sorted(fits, key=lambda f: f["speed_deg_s"])
    pick = [ex[len(ex) // 2], ex[-1]] if len(ex) > 1 else ex
    for f, col in zip(pick, (figstyle.SLOT["blue"], figstyle.SLOT["orange"])):
        ax.plot(np.array(f["_t"]) * 1000, f["_qn"], "o", ms=2.2, mfc="white", mec=col, mew=0.8)
        ax.plot(np.array(f["_t"]) * 1000, f["_fit1"], "-", color=col, lw=1.2,
                label=f"{f['speed_deg_s']:.0f}°/s: τ = {1000*f['tau_s']:.0f} ms")
        ax.plot(np.array(f["_t"]) * 1000, f["_sim07"], ":", color=col, lw=1.2)
    ax.set_xlabel("time from command onset (ms)"); ax.set_ylabel("normalised fold $q_n$")
    ax.legend(loc="lower right", fontsize=6.3); ax.set_xlim(-100, 600); figstyle.panel_label(ax, "a")
    ax.text(250, 0.25, "dotted: simulator\nlag, τ = 70 ms", fontsize=6.3, color=figstyle.INK2)
    ax = axs[1]
    sp = np.array([f["speed_deg_s"] for f in fits])
    ax.plot(sp, 1000 * taus, "o", ms=3.5, mfc="white", mec=figstyle.SLOT["blue"], mew=1.1, label="first-order fit")
    ax.axhline(70, color=figstyle.INK, lw=1.0, ls="--", label="simulator τ")
    ax.set_xscale("log"); ax.set_xlabel("commanded speed (°/s)"); ax.set_ylabel("τ (ms)"); ax.legend(fontsize=6.3)
    figstyle.panel_label(ax, "b")
    ax = axs[2]
    for tag, col in (("rest", figstyle.SLOT["blue"]), ("max_fold", figstyle.SLOT["orange"])):
        d = rep[tag]; P = np.array(d["pts"]); C = np.array(d["cubic"]); S = np.array(d["solver"])
        ax.plot(P[:, 1], -P[:, 0], "o", ms=3.0, mfc="white", mec=col, mew=1.0)
        ax.plot(C[:, 1], -C[:, 0], "-", color=col, lw=1.2, label=f"{'rest' if tag=='rest' else 'full fold'}: cubic fit")
        ax.plot(S[:, 1], -S[:, 0], ":", color=col, lw=1.2)
    ax.set_xlabel("outboard y (mm)"); ax.set_ylabel("forward −x (mm)"); ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=6.3, loc="lower left"); figstyle.panel_label(ax, "c")
    ax.text(0.98, 0.95, "dotted: energy-solver\nprediction from tip only", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.3, color=figstyle.INK2)
    fig.tight_layout(); figstyle.save(fig, os.path.join(RUNS_DIR, "paper_figures", "fig_planb_fold"))
    a = out["actuator"]; s = out["spar_shape"]
    print(f"actuator: n={a['n_steps']} tau median {1000*a['tau_median_s']:.0f} ms IQR {[round(1000*x) for x in a['tau_iqr_s']]} delay {1000*a['delay_median_s']:.0f} ms; "
          f"2nd-order wn {a['wn_median_rad_s']:.1f} rad/s zeta {a['zeta_median']:.2f}; RMSE 1st {a['rmse_first_order_median']:.3f} 2nd {a['rmse_second_order_median']:.3f} sim0.07 {a['rmse_sim_tau007_median']:.3f}")
    print("wing mode:", out["wing_mode"], "relaxation:", out["relaxation_time_constant_s"])
    print(f"spar: n={s['n_frames']} L={s['arc_length_mm_median']:.0f} mm; RMS line {s['rms_line_mm_median']:.2f} quad {s['rms_quad_mm_median']:.2f} "
          f"cubic {s['rms_cubic_mm_median']:.2f} solver {s['rms_solver_mm_median']:.2f} mm (p95 cubic {s['rms_cubic_mm_p95']:.2f}, solver {s['rms_solver_mm_p95']:.2f}); q {s['q_range']}")


if __name__ == "__main__":
    main()
