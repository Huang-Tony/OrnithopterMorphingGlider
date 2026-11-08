#!/usr/bin/env python3
"""Fig. interpretability (revision R1): morphing strategy, input sensitivity, latent structure, KAN distillation.

Reads analysis/interpretability.json (interpretability.py) and eval_final/ (teacher and KAN student), writes
paper_figures/fig_interpretability.pdf|png. Nature style (figstyle.py)."""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from common import RUNS_DIR  # noqa: E402
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

A = os.path.join(RUNS_DIR, "analysis"); FIG = os.path.join(RUNS_DIR, "paper_figures", "fig_interpretability")
VIRT = [("diff_sweep", "diff. sweep"), ("diff_span", "diff. span"), ("diff_dihedral", "diff. dihedral"),
        ("sym_sweep", "sym. sweep"), ("sym_span", "sym. span"), ("sym_dihedral", "sym. dihedral")]
GROUPS = ["reference", "body rates", "attitude", "air data", "spar shape", "tip positions", "altitude/climb"]


def slope(p, v):
    """Antisymmetric inputs: slope on r_ref (as computed); symmetric inputs: slope of the per-reference means on |r_ref|."""
    if v.startswith("diff"):
        return float(p["slope"])
    r = np.array([abs(float(k)) for k in p["by_ref"]]); y = np.array([float(val) for val in p["by_ref"].values()])
    return float(np.polyfit(r, y, 1)[0])


def main():
    figstyle.apply()
    d = json.load(open(os.path.join(A, "interpretability.json")))
    fig = plt.figure(figsize=(figstyle.DOUBLE, 4.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.62, left=0.1, right=0.985, bottom=0.08, top=0.95)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.5, 1], wspace=0.75)
    bot = outer[1].subgridspec(1, 3, wspace=0.6)
    gs = [top[0], top[1], bot[0], bot[1], bot[2]]

    # a: slope of each virtual input on r_ref (m per rad/s), per seed dots + mean bar
    ax = fig.add_subplot(gs[0]); x = np.arange(len(VIRT)); w = 0.36
    for j, (key, lab, col) in enumerate((("strategy_baseline", figstyle.LABELS["baseline"], figstyle.ENTITY["baseline"]),
                                         ("strategy_curriculum", figstyle.LABELS["curriculum"], figstyle.ENTITY["curriculum"]))):
        s = d[key]; off = (j - 0.5) * w
        per = {v: [slope(p, v) for p in s[v]["per_seed"]] for v, _ in VIRT}
        ax.bar(x + off, [np.mean(per[v]) for v, _ in VIRT], w * 0.9, color=col, alpha=0.35, lw=0, label=lab)
        for i, (v, _) in enumerate(VIRT):
            ax.plot(np.full(len(per[v]), x[i] + off), per[v], "o", ms=2.2, color=col, mew=0)
    ax.axhline(0, color=figstyle.INK, lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in VIRT], rotation=40, ha="right")
    ax.set_ylabel("use per unit yaw demand\n(m per rad/s)")
    ax.axvline(2.5, color=figstyle.LIGHT, lw=0.5)
    ax.text(1.0, 1.01, r"vs $r_\mathrm{ref}$", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5, color=figstyle.MUTED)
    ax.text(4.0, 1.01, r"vs $|r_\mathrm{ref}|$", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5, color=figstyle.MUTED)
    ax.legend(frameon=False, loc="upper right", handlelength=1.0, fontsize=6.3)
    figstyle.panel_label(ax, "a", x=-0.22, y=1.06)

    # b: normalized sensitivity per observation group (teacher)
    ax = fig.add_subplot(gs[1]); g = d["sensitivity"]["groups"]
    vals = [g.get(k, 0.0) for k in GROUPS]
    ax.barh(np.arange(len(GROUPS)), vals, color=figstyle.ENTITY["baseline"], height=0.6, lw=0)
    ax.set_yticks(np.arange(len(GROUPS))); ax.set_yticklabels(GROUPS); ax.invert_yaxis()
    ax.set_xlabel("normalized sensitivity")
    figstyle.panel_label(ax, "b", x=-0.75, y=1.06)

    # c: latent PCA explained variance + correlations of PC1/PC2
    ax = fig.add_subplot(gs[2]); ev = d["latent"]["explained_variance"]
    ax.bar(np.arange(1, len(ev) + 1), 100 * np.array(ev), color=figstyle.NEUTRAL, width=0.6, lw=0)
    ax.set_xlabel("principal component"); ax.set_ylabel("explained variance (%)")
    ax.set_xticks(np.arange(1, len(ev) + 1))
    txt = "\n".join(f"PC{k[2]} vs {v}: {d['latent'][k][v]:+.2f}" for k in ("pc1_corr", "pc2_corr")
                    for v in ("yaw_ref", "roll"))
    ax.set_ylim(0, 100 * max(ev) * 1.9)
    ax.text(0.98, 0.98, txt.replace("yaw_ref", r"$r_\mathrm{ref}$").replace("roll", r"$\phi$"), transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=figstyle.INK2)
    figstyle.panel_label(ax, "c", x=-0.42, y=1.06)

    # d: imitation loss over the DAgger iterations
    ax = fig.add_subplot(gs[3]); h = d["kan"]["history"]
    it = np.array(h["iteration"]) + 1
    ax.plot(it, h["loss"], "o-", ms=2.2, lw=0.8, color=figstyle.ENTITY["KAN"])
    ax.set_xlabel("DAgger iteration"); ax.set_ylabel("imitation loss (MSE)")
    figstyle.panel_label(ax, "d", x=-0.45, y=1.06)

    # e: final evaluation (50 episodes) teacher vs KAN student
    ax = fig.add_subplot(gs[4]); ev_dir = os.path.join(RUNS_DIR, "eval_final")
    def m(f): E = json.load(open(os.path.join(ev_dir, f)))["episodes"]; return float(np.mean([e["rms_yaw_horizon"] for e in E]))
    tch = [m(f"sac_baseline_seed{d['teacher_seed']}_{c}.json") for c in ("nominal", "randomized")]
    stu = [m(f"kan_student_{c}.json") for c in ("nominal", "randomized")]
    xx = np.arange(2)
    ax.bar(xx - 0.19, tch, 0.36, color=figstyle.ENTITY["baseline"], lw=0, label="SAC teacher")
    ax.bar(xx + 0.19, stu, 0.36, color=figstyle.ENTITY["KAN"], lw=0, label="KAN student")
    ax.set_xticks(xx); ax.set_xticklabels(["nominal", "randomized"]); ax.set_ylabel("RMS@H (rad/s)")
    k = d["kan"]
    ax.legend(frameon=False, loc="upper left", fontsize=6, handlelength=1.0)
    ax.set_ylim(0, max(tch + stu) * 1.6)
    figstyle.panel_label(ax, "e", x=-0.45, y=1.06)
    figstyle.save(fig, FIG)
    print("->", FIG)


if __name__ == "__main__":
    main()
