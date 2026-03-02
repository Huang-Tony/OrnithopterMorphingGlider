"""Visualization utilities: yaw overlay, learning curves, ablation plots."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from morphing_glider.config import _add_panel_label, _save_fig


def plot_yaw_overlay(histories, labels, title):
    if not histories: return
    yaw_ref = np.array([h["yaw_ref"] for h in histories[0]])
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(yaw_ref, lw=2.5, ls="--", color="black", label="Yaw Reference")
    for hist, lab in zip(histories, labels):
        yr = np.array([h["yaw_rate"] for h in hist])
        ax.plot(yr, lw=1.8, label=f"yaw_rate ({lab})")
    ax.set_xlabel("Step"); ax.set_ylabel("Yaw Rate (rad/s)"); ax.set_title(title)
    ax.grid(True, alpha=0.2); ax.legend(); _add_panel_label(ax, "A")
    plt.tight_layout(); _save_fig(fig, "yaw_overlay.png", "Yaw tracking comparison across controllers")
    plt.show()


def plot_learning_curves(all_logs, title):
    if not all_logs:
        print("[LearningCurves] No logs; skipping."); return
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    for i, (algo, logs) in enumerate(sorted(all_logs.items())):
        trace = sorted(logs.get("evaltrace", []), key=lambda d: int(d.get("global_steps", 0)))
        if not trace: continue
        steps = np.array([int(d["global_steps"]) for d in trace])
        mean = np.array([float(d["mean_rmsh"]) for d in trace])
        lo = np.array([float(d["lo_rmsh"]) for d in trace])
        hi = np.array([float(d["hi_rmsh"]) for d in trace])
        c = colors[i % len(colors)]
        ax.plot(steps, mean, lw=2, label=str(algo), color=c)
        ax.fill_between(steps, lo, hi, color=c, alpha=0.15)
        for b in logs.get("phase_boundaries", []):
            x = int(b.get("global_steps", -1))
            if x >= 0: ax.axvline(x, color=c, ls="--", lw=0.8, alpha=0.2)
    ax.set_xlabel("Environment Steps"); ax.set_ylabel("Yaw RMS@H (rad/s)")
    ax.set_title(title); ax.grid(True, alpha=0.2); ax.legend()
    _add_panel_label(ax, "A")
    plt.tight_layout(); _save_fig(fig, "learning_curves.png", "Learning curves with BCa bootstrap CIs")
    plt.show()


def generate_animation(history, *, stride=1, interval_ms=50):
    stride = max(1, int(stride))
    idx = list(range(0, len(history), stride))
    yr = np.array([history[i]["yaw_rate"] for i in range(len(history))])
    yref = np.array([history[i]["yaw_ref"] for i in range(len(history))])
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(yr, lw=2, label="yaw_rate"); ax.plot(yref, lw=2, ls="--", label="yaw_ref")
    vline = ax.axvline(0, color="k", alpha=0.4); ax.legend(); ax.grid(True, alpha=0.2)
    def update(fi): vline.set_xdata([idx[fi], idx[fi]]); return (vline,)
    anim = animation.FuncAnimation(fig, update, frames=len(idx), interval=int(interval_ms))
    plt.close(fig)
    return anim


def plot_ablation_summary(ablation_results, save_path="ablation_summary.png"):
    if not ablation_results:
        print("[Ablation] No results to plot"); return
    conditions = sorted(ablation_results.keys())
    metrics_to_plot = ["rms_yaw_steady", "failure", "mean_settle_time"]
    n_met = len(metrics_to_plot); n_cond = len(conditions)
    fig, axes = plt.subplots(1, n_met, figsize=(4 * n_met, 5))
    if n_met == 1: axes = [axes]
    for mi, mk in enumerate(metrics_to_plot):
        ax = axes[mi]; means = []; errs_lo = []; errs_hi = []; xlabels = []
        for cond in conditions:
            s = ablation_results[cond].get("summaries", {}).get(mk, {})
            m = float(s.get("mean", np.nan)); lo = float(s.get("lo", np.nan)); hi = float(s.get("hi", np.nan))
            means.append(m); errs_lo.append(max(0, m - lo)); errs_hi.append(max(0, hi - m))
            xlabels.append(cond.replace("_", "\n"))
        x = np.arange(n_cond)
        ax.bar(x, means, yerr=[errs_lo, errs_hi], capsize=3, alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=6, rotation=45, ha='right')
        ax.set_ylabel(mk); ax.set_title(mk); ax.grid(True, alpha=0.2, axis='y')
    _add_panel_label(axes[0], "A")
    plt.tight_layout(); _save_fig(fig, save_path, "Ablation suite: effect of each design choice")
    plt.show()
