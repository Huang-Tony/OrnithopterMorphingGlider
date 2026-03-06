"""Visualization utilities: yaw overlay, learning curves, ablation plots."""

import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from morphing_glider.config import _add_panel_label, _save_fig


def plot_yaw_overlay(histories, labels, title):
    if not histories: return
    # Use the longest history's yaw_ref so the reference spans the full episode
    longest_hist = max(histories, key=len)
    yaw_ref = np.array([h["yaw_ref"] for h in longest_hist])
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


# ================================================================
# Per-controller yaw tracking grid
# ================================================================

def plot_yaw_overlay_grid(histories, labels, title="Per-controller yaw tracking"):
    """Grid of subplots showing each controller's yaw tracking individually."""
    n = len(histories)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    # Use longest history's yaw_ref as the canonical reference
    ref_hist = max(histories, key=len)
    yaw_ref = np.array([h["yaw_ref"] for h in ref_hist])

    for i, (hist, lab) in enumerate(zip(histories, labels)):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        yr = np.array([h["yaw_rate"] for h in hist])
        ref_i = np.array([h["yaw_ref"] for h in hist])
        ax.plot(ref_i, lw=2, ls="--", color="black", alpha=0.6, label="Reference")
        ax.plot(yr, lw=1.3, color="C0", label="Actual")
        err = np.abs(yr - ref_i[:len(yr)])
        rms_val = float(np.sqrt(np.mean(err ** 2)))
        ep_len = len(hist)
        ax.set_title(f"{lab}  (RMS={rms_val:.3f}, T={ep_len})", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Yaw Rate (rad/s)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, "yaw_overlay_grid.png", "Per-controller yaw tracking grid")
    plt.show()


# ================================================================
# Attitude stability comparison
# ================================================================

def plot_attitude_stability(histories, labels, save_path="attitude_stability.png"):
    """Plot roll and pitch time series for multiple controllers."""
    if not histories:
        return
    fig, (ax_roll, ax_pitch) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for hist, lab in zip(histories, labels):
        roll = np.degrees(np.array([h["roll"] for h in hist]))
        pitch = np.degrees(np.array([h["pitch"] for h in hist]))
        ax_roll.plot(roll, lw=1.0, label=lab, alpha=0.8)
        ax_pitch.plot(pitch, lw=1.0, label=lab, alpha=0.8)

    ax_roll.set_ylabel("Roll (deg)")
    ax_roll.set_title("Roll Stability")
    ax_roll.legend(fontsize=7, ncol=2, loc="upper right")
    ax_roll.grid(True, alpha=0.2)
    ax_pitch.set_ylabel("Pitch (deg)")
    ax_pitch.set_title("Pitch Stability")
    ax_pitch.set_xlabel("Step")
    ax_pitch.legend(fontsize=7, ncol=2, loc="upper right")
    ax_pitch.grid(True, alpha=0.2)
    _add_panel_label(ax_roll, "A")
    _add_panel_label(ax_pitch, "B")
    plt.tight_layout()
    _save_fig(fig, save_path, "Roll and pitch stability across controllers")
    plt.show()


# ================================================================
# Wing morphing action decomposition
# ================================================================

def plot_action_decomposition(history, *, label="Agent",
                               save_path="morphing_glider_figures/action_decomposition.png"):
    """Plot the 6 action channels over time for a single episode."""
    if not history:
        return
    actions = np.array([h["action"] for h in history])
    yaw_ref = np.array([h["yaw_ref"] for h in history])
    channel_names = [r"$\Delta x_R$", r"$\Delta y_R$", r"$\Delta z_R$",
                     r"$\Delta x_L$", r"$\Delta y_L$", r"$\Delta z_L$"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for i, (ax, name) in enumerate(zip(axes.flat, channel_names)):
        ax.plot(actions[:, i], lw=1.0, color="C0")
        ax_tw = ax.twinx()
        ax_tw.plot(yaw_ref, lw=0.8, ls="--", color="gray", alpha=0.4)
        ax_tw.set_ylabel("ref", fontsize=7, color="gray")
        ax_tw.tick_params(axis="y", labelsize=7, colors="gray")
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel("Step")
    axes[-1, 1].set_xlabel("Step")
    fig.suptitle(f"Wing Morphing Action Channels: {label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, save_path, f"Action decomposition for {label}")
    plt.show()


# ================================================================
# Training loss / gradient curves from TensorBoard
# ================================================================

def plot_training_losses(tb_dirs, algo_labels, save_path="training_losses.png"):
    """Plot actor loss, critic loss, entropy coeff, and episode reward from TB logs."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[TrainingLosses] tensorboard not available; skipping.")
        return

    tags_to_plot = [
        ("train/actor_loss", "Actor Loss"),
        ("train/critic_loss", "Critic Loss"),
        ("train/ent_coef", "Entropy Coefficient"),
        ("rollout/ep_rew_mean", "Mean Episode Reward"),
    ]

    fig, axes_2d = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes_2d.flat
    panel_labels = ["A", "B", "C", "D"]

    for tb_dir, label in zip(tb_dirs, algo_labels):
        if not os.path.isdir(tb_dir):
            continue
        # SB3 creates subdirs like SAC_1/, SAC_2/; use the latest
        subdirs = sorted(
            [os.path.join(tb_dir, d) for d in os.listdir(tb_dir)
             if os.path.isdir(os.path.join(tb_dir, d))],
            key=os.path.getmtime)
        tb_path = subdirs[-1] if subdirs else tb_dir

        try:
            ea = EventAccumulator(tb_path)
            ea.Reload()
            available = ea.Tags().get("scalars", [])
        except Exception as exc:
            print(f"[TrainingLosses] Could not load {tb_path}: {exc!r}")
            continue

        for (tag, _title), ax in zip(tags_to_plot, axes_flat):
            if tag in available:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                ax.plot(steps, values, label=label, alpha=0.7, lw=1)

    for idx, ((tag, title), ax) in enumerate(zip(tags_to_plot, axes_flat)):
        ax.set_xlabel("Steps")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        _add_panel_label(ax, panel_labels[idx])

    plt.tight_layout()
    _save_fig(fig, save_path, "Training loss and reward curves (SAC)")
    plt.show()


# ================================================================
# Performance comparison bar chart
# ================================================================

def plot_performance_comparison(eval_blocks, *, condition="nominal",
                                save_path="performance_comparison.png"):
    """Grouped bar chart comparing key metrics across all evaluated controllers."""
    if not eval_blocks:
        return

    metrics = [
        ("rms_yaw_horizon", "RMS@H (rad/s)", False),
        ("failure", "Failure Rate (%)", True),
        ("mean_settle_time", "Settle Time (steps)", False),
        ("mean_action_norm", "Action Norm", False),
    ]

    agents = [(algo, block) for algo, cond, block in eval_blocks if cond == condition]
    if not agents:
        return

    n_metrics = len(metrics)
    n_agents = len(agents)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_agents)
    xlabels = [a.replace("_", "\n") for a, _ in agents]
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_agents, 3)))

    for mi, (key, ylabel, is_rate) in enumerate(metrics):
        ax = axes[mi]
        means = []
        lo_err = []
        hi_err = []
        for _, block in agents:
            s = block.get("summaries", {}).get(key, {})
            m = float(s.get("mean", np.nan))
            l = float(s.get("lo", np.nan))
            h = float(s.get("hi", np.nan))
            if is_rate:
                m *= 100
                l *= 100
                h *= 100
            means.append(m)
            lo_err.append(max(0, m - l) if np.isfinite(l) else 0)
            hi_err.append(max(0, h - m) if np.isfinite(h) else 0)

        ax.bar(x, means, yerr=[lo_err, hi_err], capsize=3, alpha=0.85,
               color=colors[:n_agents])
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.2, axis="y")

    _add_panel_label(axes[0], "A")
    fig.suptitle(f"Performance Comparison ({condition})", fontsize=13, fontweight="bold",
                 y=1.02)
    plt.tight_layout()
    _save_fig(fig, save_path, f"Performance comparison ({condition})")
    plt.show()
