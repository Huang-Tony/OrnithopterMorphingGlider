#!/usr/bin/env python3
"""Control-law readout of the distilled KAN student (revision R1).

Read out on the states that the KAN student visits in its own closed-loop nominal evaluation episodes
(eval_final/kan_student_nominal.npz). The SAC teacher is queried (deterministically) on exactly the same states, which
are also the kind of states on which DAgger trains the student, so that every bin compares the two control laws on
identical inputs. Read out: the commanded differential sweep 0.5 (a_Rx - a_Lx) in bins of the yaw-rate error
e = r_ref - r for each reference r_ref in {-0.6, -0.3, 0, 0.3, 0.6} rad/s (bins with fewer than 15 samples omitted),
and the mean symmetric tip raise 0.5 (a_Rz + a_Lz) per reference.
Output: analysis/kan_readout.json, paper_figures/fig_kan.pdf|png
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from common import RUNS_DIR, dump_json  # noqa: E402
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from morphing_glider.environment.observation import OBS_IDX  # noqa: E402

REFS = [-0.6, -0.3, 0.0, 0.3, 0.6]


def binned(O, A, rref_val, edges):
    rr = O[..., OBS_IDX["yaw_ref"]]; r = O[..., OBS_IDX["omega_r"]]
    ok = np.isfinite(A[..., 0]) & (np.abs(rr - rref_val) < 1e-6)
    e = (rr - r)[ok]; ds = (0.5 * (A[..., 0] - A[..., 3]))[ok]
    idx = np.digitize(e, edges) - 1; m, c = [], []
    for k in range(len(edges) - 1):
        sel = idx == k; m.append(float(np.mean(ds[sel])) if sel.sum() >= 15 else None); c.append(int(sel.sum()))
    return m, c


def tip_raise_by_ref(O, A):
    rr = O[..., OBS_IDX["yaw_ref"]]; ok = np.isfinite(A[..., 0]); sz = 0.5 * (A[..., 2] + A[..., 5])
    return {str(v): float(np.mean(sz[ok & (np.abs(rr - v) < 1e-6)])) for v in REFS}


def teacher_on(O, model_path):
    """Deterministic teacher actions on the given raw observations (same normalization as in evaluation)."""
    from evaluate import load_sac
    ctrl = load_sac(model_path).ctrl; A = np.full(O.shape[:-1] + (6,), np.nan, dtype=np.float32)
    ok = np.all(np.isfinite(O), axis=-1)
    obs_n = ctrl._normalize_obs(O[ok].astype(np.float32))
    act, _ = ctrl.model.predict(obs_n, deterministic=True)
    A[ok] = act
    return A


def main():
    """On-distribution readout from the closed-loop evaluation episodes (nominal) of the KAN student and its teacher."""
    it = json.load(open(os.path.join(RUNS_DIR, "analysis", "interpretability.json"))); seed = int(it["teacher_seed"])
    EV = os.path.join(RUNS_DIR, "eval_final")
    st = np.load(os.path.join(EV, "kan_student_nominal.npz")); O = st["obs"].astype(np.float64)
    acts = {"student": st["action"], "teacher": teacher_on(O, os.path.join(RUNS_DIR, "models", f"baseline_seed{seed}.zip"))}
    edges = np.linspace(-0.6, 0.6, 13); mid = 0.5 * (edges[1:] + edges[:-1])
    out = {"doc": __doc__, "teacher_seed": seed, "bin_mid": mid.tolist(), "student": {}, "teacher": {}}
    for name, A in acts.items():
        for rr in REFS:
            m, c = binned(O, A, rr, edges); out[name][str(rr)] = {"diff_sweep": m, "n": c}
        out[name]["tip_raise"] = tip_raise_by_ref(O, A)
    devs = [abs(a - b) for rr in REFS for a, b in zip(out["student"][str(rr)]["diff_sweep"], out["teacher"][str(rr)]["diff_sweep"])
            if a is not None and b is not None]
    out["mean_abs_student_teacher_diff_sweep_m"] = float(np.mean(devs)) if devs else None
    out["student_tip_raise_turn_minus_straight_m"] = float(0.5 * (out["student"]["tip_raise"]["0.6"] + out["student"]["tip_raise"]["-0.6"])
                                                           - out["student"]["tip_raise"]["0.0"])
    out["teacher_tip_raise_turn_minus_straight_m"] = float(0.5 * (out["teacher"]["tip_raise"]["0.6"] + out["teacher"]["tip_raise"]["-0.6"])
                                                           - out["teacher"]["tip_raise"]["0.0"])
    for name in ("student", "teacher"):          # pooled small-error gain: least-squares slope of bin means, |e| <= 0.3
        xs, ys = [], []
        for rr in REFS:
            for xm, v in zip(mid, out[name][str(rr)]["diff_sweep"]):
                if v is not None and abs(xm) <= 0.3: xs.append(xm); ys.append(v)
        out[f"{name}_gain_m_per_rad_s"] = float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 3 else None
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "kan_readout.json"))

    figstyle.apply()
    fig, axs = plt.subplots(1, 2, figsize=(figstyle.DOUBLE, 2.2), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.38))
    cmap = plt.get_cmap("coolwarm"); ax = axs[0]
    for i, rr in enumerate(REFS):
        col = cmap(i / (len(REFS) - 1))
        for name, ls, lw in (("student", "-", 1.0), ("teacher", ":", 0.8)):
            y = np.array([np.nan if v is None else v for v in out[name][str(rr)]["diff_sweep"]]); ok = np.isfinite(y)
            ax.plot(mid[ok], y[ok], ls, color=col, lw=lw, marker="o" if name == "student" else None, ms=2.0,   # sparse bins skipped, not gapped
                    label=f"{rr:+.1f}".replace("-", "\u2212") if name == "student" else None)
    ax.axhline(0, color=figstyle.LIGHT, lw=0.5); ax.axvline(0, color=figstyle.LIGHT, lw=0.5)
    ax.set_xlabel(r"yaw-rate error $e=r_\mathrm{ref}-r$ (rad/s)"); ax.set_ylabel("differential sweep (m)")
    ax.legend(title=r"$r_\mathrm{ref}$ (rad/s)", fontsize=6, title_fontsize=6, loc="lower right", frameon=False, ncol=2,
              columnspacing=0.8, handlelength=1.4)
    ax.text(0.02, 0.98, "solid: KAN student\ndotted: SAC teacher, same states", transform=ax.transAxes, fontsize=6, va="top",
            color=figstyle.INK2)
    figstyle.panel_label(ax, "a", x=-0.2)
    ax = axs[1]; x = np.arange(len(REFS))
    ax.bar(x - 0.19, [out["teacher"]["tip_raise"][str(v)] for v in REFS], 0.36, color=figstyle.ENTITY["baseline"], lw=0, label="SAC teacher")
    ax.bar(x + 0.19, [out["student"]["tip_raise"][str(v)] for v in REFS], 0.36, color=figstyle.ENTITY["KAN"], lw=0, label="KAN student")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:+.1f}".replace("-", "\u2212") for v in REFS]); ax.set_xlabel(r"$r_\mathrm{ref}$ (rad/s)")
    ax.set_ylabel("mean symmetric tip raise (m)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6, borderaxespad=0.2,
              handlelength=1.0, handleheight=0.8)
    figstyle.panel_label(ax, "b", x=-0.3)
    figstyle.save(fig, os.path.join(RUNS_DIR, "paper_figures", "fig_kan"))
    print({k: out[k] for k in ("mean_abs_student_teacher_diff_sweep_m", "student_tip_raise_turn_minus_straight_m",
                               "teacher_tip_raise_turn_minus_straight_m")}, out["student"]["tip_raise"], out["teacher"]["tip_raise"])


if __name__ == "__main__":
    main()
