#!/usr/bin/env python3
"""Statistics, number macros and Nature-style result figures from revision_runs/eval_* (works with partial data).

Statistics: SAC metrics use a hierarchical bootstrap (seeds outer, episodes inner); classical controllers a
BCa bootstrap over episodes; SAC-vs-classical comparisons pair episodes by evaluation seed (Wilcoxon signed-
rank, pooled over training seeds); algorithm comparisons use seed-level means (exact Wilcoxon, Holm) together
with bootstrap intervals of the difference, following the interval-estimate recommendations of Agarwal et al.
Outputs: analysis/results_summary.json (incl. 'macros' consumed by make_numbers.py) and paper_figures/*.
"""

import glob
import json
import math
import os

import numpy as np
from scipy import stats

import common  # noqa: F401
from common import RUNS_DIR, TRAIN_SEEDS, CONDITIONS, algo_seed, dump_json
from morphing_glider.config import DT
import figstyle
import matplotlib.pyplot as plt

EV = os.path.join(RUNS_DIR, "eval_final"); RB = os.path.join(RUNS_DIR, "eval_robust"); FIG = os.path.join(RUNS_DIR, "paper_figures")
ALGOS = ["baseline", "curriculum", "residual_curriculum"]
CLASSICAL = ["Alloc-ML", "Heuristic", "PID", "GS-PID", "LQR", "MPC", "Zero"]
METRICS = ["rms_yaw_horizon", "rms_yaw_steady", "rms_yaw_transient", "failure", "mean_settle_time", "rms_roll_deg",
           "rms_pitch_deg", "max_abs_roll_deg", "rms_roll_rate_dps", "mean_tip_speed_mps", "mean_sq_action_increment",
           "hf_action_power_frac_2Hz", "mean_tip_displacement_m", "mean_abs_dihedral_m", "mean_struct_energy_norm",
           "mean_aero_power_W", "mean_energy_loss_rate_W", "mean_sink_rate_mps", "mean_airspeed_mps", "altitude_loss_m",
           "survival_time_s", "rms_yaw_survived", "mean_ctrl_latency_ms", "mean_action_norm"]
RNG = np.random.default_rng(2026)


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def ep_values(eps, k):
    v = np.array([e.get(k, np.nan) for e in eps], float)
    if k == "mean_settle_time": v = v * DT                      # steps -> seconds
    return v


def boot_ci(x, n=4000):
    x = x[np.isfinite(x)]
    if x.size == 0: return (np.nan, np.nan, np.nan)
    if x.size == 1 or np.allclose(x, x[0]): return (float(x.mean()),) * 3
    res = stats.bootstrap((x,), np.mean, n_resamples=n, method="BCa", random_state=RNG)
    return float(x.mean()), float(res.confidence_interval.low), float(res.confidence_interval.high)


def hier_ci(per_seed, n=4000):
    per_seed = [p[np.isfinite(p)] for p in per_seed if np.isfinite(p).any()]
    if not per_seed: return (np.nan, np.nan, np.nan)
    m = float(np.mean([p.mean() for p in per_seed])); bs = []
    for _ in range(n):
        idx = RNG.integers(0, len(per_seed), len(per_seed))
        bs.append(np.mean([RNG.choice(per_seed[i], per_seed[i].size).mean() for i in idx]))
    return m, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def collect():
    data = {}
    for algo in ALGOS:
        for cond in CONDITIONS:
            seeds = {}
            for s in TRAIN_SEEDS:
                d = load(os.path.join(EV, f"sac_{algo}_seed{algo_seed(algo, s)}_{cond}.json"))
                if d: seeds[s] = d["episodes"]
            if seeds: data[(algo, cond)] = seeds
    for c in CLASSICAL + ["KAN student"]:
        for cond in CONDITIONS:
            f = f"kan_student_{cond}.json" if c == "KAN student" else f"classical_{c}_{cond}.json"
            d = load(os.path.join(EV, f))
            if d: data[(c, cond)] = {0: d["episodes"]}
    return data


def summarise(data):
    S = {}
    for (name, cond), seeds in data.items():
        row = {"n_seeds": len(seeds), "n_episodes": int(sum(len(v) for v in seeds.values()))}
        for k in METRICS:
            per = [ep_values(v, k) for v in seeds.values()]
            row[k] = hier_ci(per) if name in ALGOS else boot_ci(np.concatenate(per))
        S[f"{name}|{cond}"] = row
    return S


def paired_vs(data, a, b, cond, k="rms_yaw_horizon"):
    """Pair episodes by evaluation seed (episode index) of a (pooled over training seeds) with b."""
    if (a, cond) not in data or (b, cond) not in data: return None
    bv = {e["seed"]: e[k] for e in data[(b, cond)][0]}
    xs, ys = [], []
    for eps in data[(a, cond)].values():
        for e in eps:
            if e["seed"] in bv: xs.append(e[k]); ys.append(bv[e["seed"]])
    xs, ys = np.array(xs), np.array(ys)
    if xs.size < 5: return None
    w = stats.wilcoxon(xs, ys)
    diff = xs - ys; d = float(diff.mean() / max(diff.std(ddof=1), 1e-12))
    return {"n_pairs": int(xs.size), "p_wilcoxon": float(w.pvalue), "mean_diff": float(diff.mean()), "d_paired": d,
            "ratio": float(xs.mean() / ys.mean())}


def seed_level(data, a, b, cond, k="rms_yaw_horizon"):
    if (a, cond) not in data or (b, cond) not in data: return None
    xa = np.array([np.mean([e[k] for e in v]) for v in data[(a, cond)].values()])
    xb = np.array([np.mean([e[k] for e in v]) for v in data[(b, cond)].values()])
    out = {"n_a": int(xa.size), "n_b": int(xb.size), "mean_a": float(xa.mean()), "mean_b": float(xb.mean())}
    if xa.size >= 2 and xb.size >= 2:
        out["p_mannwhitney"] = float(stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
        bs = [RNG.choice(xa, xa.size).mean() - RNG.choice(xb, xb.size).mean() for _ in range(4000)]
        out["diff_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        sp = math.sqrt((xa.var(ddof=1) + xb.var(ddof=1)) / 2); out["cohen_d"] = float((xa.mean() - xb.mean()) / max(sp, 1e-12))
    return out


def best_classical(S, cond):
    c = [(S[f"{n}|{cond}"]["rms_yaw_horizon"][0], n) for n in CLASSICAL if f"{n}|{cond}" in S]
    return min(c)[1] if c else None


# ------------------------------------------------------------------ figures
def label(n): return figstyle.LABELS.get(n, n)


def color(n, i=0):
    return figstyle.ENTITY.get(n, figstyle.CLASSICAL_GREYS[i % len(figstyle.CLASSICAL_GREYS)])


def fig_performance(S):
    names = [n for n in ALGOS + ["KAN student"] + CLASSICAL if any(f"{n}|{c}" in S for c in CONDITIONS)]
    mets = [("rms_yaw_horizon", "RMS@H (rad/s)", 1), ("failure", "failure rate (%)", 100), ("mean_settle_time", "settling time (s)", 1)]
    fig, axs = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 0.32 * len(names) + 0.55), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, (k, xl, sc) in zip(axs, mets):
        for i, n in enumerate(names):
            for cond, off, mfc in (("nominal", 0.12, None), ("randomized", -0.12, "white")):
                key = f"{n}|{cond}"
                if key not in S: continue
                m, lo, hi = (v * sc for v in S[key][k]); col = color(n, i)
                ax.plot([lo, hi], [y[i] + off] * 2, color=col, lw=0.8, solid_capstyle="butt")
                ax.plot(m, y[i] + off, "o", ms=3.2, mec=col, mfc=mfc or col, mew=0.8)
        ax.set_xlabel(xl); ax.set_xlim(left=0)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(y); axs[0].set_yticklabels([label(n) for n in names])
    axs[2].plot([], [], "o", ms=3.2, color=figstyle.INK, label="nominal"); axs[2].plot([], [], "o", ms=3.2, mfc="white", mec=figstyle.INK, label="randomized")
    axs[2].legend(loc="lower right")
    for ax, l in zip(axs, "abc"): figstyle.panel_label(ax, l, x=-0.05 if l != "a" else -0.42)
    fig.tight_layout(w_pad=1.0); figstyle.save(fig, os.path.join(FIG, "fig_performance"))


def fig_extended(S):
    names = [n for n in ALGOS + ["KAN student"] + CLASSICAL if f"{n}|nominal" in S]
    mets = [("rms_roll_deg", "RMS roll (deg)"), ("rms_pitch_deg", "RMS pitch (deg)"), ("mean_tip_speed_mps", "tip speed (m/s)"),
            ("mean_sq_action_increment", "action increment (m$^2$)"), ("mean_tip_displacement_m", "tip displacement (m)"),
            ("mean_energy_loss_rate_W", "energy loss rate (W)"), ("mean_sink_rate_mps", "sink rate (m/s)"),
            ("mean_ctrl_latency_ms", "controller time (ms)")]
    fig, axs = plt.subplots(2, 4, figsize=(figstyle.DOUBLE, 0.24 * len(names) * 2 + 1.0), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, (k, xl), l in zip(axs.flat, mets, "abcdefgh"):
        for i, n in enumerate(names):
            m, lo, hi = S[f"{n}|nominal"][k]; col = color(n, i)
            ax.plot([lo, hi], [y[i]] * 2, color=col, lw=0.8); ax.plot(m, y[i], "o", ms=3.0, color=col)
        ax.set_xlabel(xl); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        if k == "mean_ctrl_latency_ms": ax.set_xscale("log")
        figstyle.panel_label(ax, l, x=-0.08)
    for r in range(2): axs[r, 0].set_yticks(y); axs[r, 0].set_yticklabels([label(n) for n in names])
    fig.tight_layout(w_pad=0.8, h_pad=1.0); figstyle.save(fig, os.path.join(FIG, "fig_extended_metrics"))


def fig_timehistory(data, cond="nominal"):
    best = best_classical_cache.get(cond)
    if ("baseline", cond) not in data or not best: return
    seed7 = sorted(data[("baseline", cond)].keys())[0]
    f_sac = os.path.join(EV, f"sac_baseline_seed{seed7}_{cond}.npz"); f_cl = os.path.join(EV, f"classical_{best}_{cond}.npz")
    if not (os.path.exists(f_sac) and os.path.exists(f_cl)): return
    A, B = np.load(f_sac), np.load(f_cl)
    # representative episode: median RMS@H of the SAC policy
    eps = data[("baseline", cond)][seed7]; order = np.argsort([e["rms_yaw_horizon"] for e in eps]); j = int(order[len(order) // 2])
    t = np.arange(A["yaw_rate"].shape[1]) * DT
    fig, axs = plt.subplots(3, 1, figsize=(figstyle.SINGLE * 1.25, 3.6), sharex=True)
    ax = axs[0]; ax.step(t, A["yaw_ref"][j], where="post", color=figstyle.INK, lw=0.8, ls="--", label="reference")
    ax.plot(t, A["yaw_rate"][j], color=figstyle.SLOT["blue"], lw=1.0, label="baseline SAC")
    ax.plot(t, B["yaw_rate"][j], color=figstyle.CLASSICAL_GREYS[1], lw=1.0, label=label(best))
    ax.set_ylabel("yaw rate (rad/s)"); ax.legend(loc="upper right", ncol=3, fontsize=5.6); figstyle.panel_label(ax, "a", x=-0.14)
    ax = axs[1]
    ax.plot(t, np.degrees(A["roll"][j]), color=figstyle.SLOT["blue"], lw=1.0, label="roll"); ax.plot(t, np.degrees(A["pitch"][j]), color=figstyle.SLOT["blue"], lw=0.8, ls=":", label="pitch")
    ax.plot(t, np.degrees(B["roll"][j]), color=figstyle.CLASSICAL_GREYS[1], lw=1.0); ax.plot(t, np.degrees(B["pitch"][j]), color=figstyle.CLASSICAL_GREYS[1], lw=0.8, ls=":")
    ax.set_ylabel("attitude (deg)"); ax.legend(loc="upper right", ncol=2, fontsize=5.6); figstyle.panel_label(ax, "b", x=-0.14)
    ax = axs[2]
    from interpretability import virtual
    V = virtual(A["action"][j])
    for c, nm, col in ((0, "diff. sweep", figstyle.SLOT["blue"]), (2, "diff. dihedral", figstyle.SLOT["orange"]), (4, "diff. span", figstyle.SLOT["aqua"]), (1, "sym. sweep", figstyle.SLOT["violet"])):
        ax.plot(t, V[:, c], color=col, lw=0.9, label=nm)
    ax.set_ylabel("command (m)"); ax.set_xlabel("time (s)"); ax.legend(loc="upper right", ncol=4, fontsize=5.4); figstyle.panel_label(ax, "c", x=-0.14)
    fig.tight_layout(h_pad=0.4); figstyle.save(fig, os.path.join(FIG, "fig_timehistory"))


def fig_learning():
    fig, axs = plt.subplots(1, 3, figsize=(figstyle.DOUBLE, 2.0))
    for algo in ALGOS:
        curves, ent, std = [], [], []
        for s in TRAIN_SEEDS:
            d = load(os.path.join(RUNS_DIR, "logs", f"{algo}_seed{algo_seed(algo, s)}.train.json"))
            if not d: continue
            et = d["logs"]["evaltrace"]; curves.append((np.array([e["global_steps"] for e in et]), np.array([e["mean_rmsh"] for e in et])))
            dg = d["diagnostics"]; ent.append((np.array([r["t"] for r in dg]), np.array([r.get("ent_coef", np.nan) for r in dg])))
            std.append((np.array([r["t"] for r in dg]), np.array([r.get("policy_std_mean", np.nan) for r in dg])))
        if not curves: continue
        col = figstyle.ENTITY[algo]
        for ax, cs in zip(axs, (curves, ent, std)):
            grid = np.linspace(0, 1e6, 41); Y = np.array([np.interp(grid, x, y, left=np.nan, right=np.nan) for x, y in cs])
            ax.plot(grid / 1e6, np.nanmean(Y, 0), color=col, lw=1.0, label=label(algo))
            if Y.shape[0] > 1: ax.fill_between(grid / 1e6, np.nanmin(Y, 0), np.nanmax(Y, 0), color=col, alpha=0.15, lw=0)
    axs[0].set_ylabel("RMS@H (rad/s)"); axs[1].set_ylabel(r"entropy coefficient $\alpha$"); axs[2].set_ylabel("policy std (pre-squash)")
    axs[1].set_yscale("log")
    for ax, l in zip(axs, "abc"): ax.set_xlabel("environment steps (M)"); figstyle.panel_label(ax, l, x=-0.24)
    axs[0].legend(loc="upper right", fontsize=5.6)
    fig.tight_layout(w_pad=1.5); figstyle.save(fig, os.path.join(FIG, "fig_learning"))


best_classical_cache = {}


def main():
    figstyle.apply()
    data = collect(); S = summarise(data)
    for cond in CONDITIONS: best_classical_cache[cond] = best_classical(S, cond)
    tests = {}
    for cond in CONDITIONS:
        bc = best_classical_cache[cond]
        tests[f"baseline_vs_best_classical|{cond}"] = {"best": bc, **(paired_vs(data, "baseline", bc, cond) or {})}
        for other in ("curriculum", "residual_curriculum"):
            tests[f"baseline_vs_{other}|{cond}"] = seed_level(data, "baseline", other, cond)
        tests[f"baseline_vs_kan|{cond}"] = paired_vs(data, "baseline", "KAN student", cond)
    out = {"summary": S, "tests": tests, "best_classical": best_classical_cache}
    # macros for the manuscript
    mac = {}
    def fm(v, f="{:.3f}"): return "\\textbf{??}" if v is None or not np.isfinite(v) else f.format(v)
    tag = {"baseline": "Base", "curriculum": "Curr", "residual_curriculum": "Resid", "KAN student": "Kan", "Alloc-ML": "Alloc",
           "Heuristic": "Heur", "PID": "Pid", "GS-PID": "Gspid", "LQR": "Lqr", "MPC": "Mpc", "Zero": "Zero"}
    for key, row in S.items():
        n, cond = key.split("|"); c = "Nom" if cond == "nominal" else "Rand"
        for k, short, f, sc in (("rms_yaw_horizon", "Rmsh", "{:.3f}", 1), ("rms_yaw_steady", "Rmsss", "{:.3f}", 1), ("failure", "Fail", "{:.0f}", 100),
                                ("mean_settle_time", "Settle", "{:.2f}", 1), ("rms_roll_deg", "Roll", "{:.1f}", 1), ("mean_energy_loss_rate_W", "Energy", "{:.2f}", 1),
                                ("mean_sink_rate_mps", "Sink", "{:.2f}", 1), ("mean_tip_speed_mps", "TipSpeed", "{:.3f}", 1),
                                ("mean_tip_displacement_m", "TipDisp", "{:.3f}", 1), ("mean_ctrl_latency_ms", "Lat", "{:.2f}", 1),
                                ("mean_sq_action_increment", "Jerk", "{:.4f}", 1), ("rms_yaw_survived", "RmsSurv", "{:.3f}", 1)):
            m, lo, hi = row[k]
            mac[f"R{tag[n]}{c}{short}"] = fm(m * sc, f); mac[f"R{tag[n]}{c}{short}Lo"] = fm(lo * sc, f); mac[f"R{tag[n]}{c}{short}Hi"] = fm(hi * sc, f)
    for cond in CONDITIONS:
        c = "Nom" if cond == "nominal" else "Rand"; t = tests[f"baseline_vs_best_classical|{cond}"]
        mac[f"RBestClassical{c}"] = str(t.get("best"))
        mac[f"RRatioBest{c}"] = fm(t.get("ratio", np.nan), "{:.2f}"); mac[f"RPBest{c}"] = fm(t.get("p_wilcoxon", np.nan), "{:.1e}")
        for other, o in (("curriculum", "Curr"), ("residual_curriculum", "Resid")):
            t2 = tests.get(f"baseline_vs_{other}|{cond}") or {}
            mac[f"RPBaseVs{o}{c}"] = fm(t2.get("p_mannwhitney", np.nan), "{:.2f}"); mac[f"RDBaseVs{o}{c}"] = fm(t2.get("cohen_d", np.nan), "{:.2f}")
    out["macros"] = mac
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "results_summary.json"))
    fig_performance(S); fig_extended(S); fig_timehistory(data); fig_learning()
    for key in sorted(S):
        r = S[key]; print(f"{key:<30} RMS@H {r['rms_yaw_horizon'][0]:.3f} [{r['rms_yaw_horizon'][1]:.3f},{r['rms_yaw_horizon'][2]:.3f}] "
                          f"fail {100*r['failure'][0]:.0f}% settle {r['mean_settle_time'][0]:.2f}s roll {r['rms_roll_deg'][0]:.1f}deg (seeds {r['n_seeds']})")
    print(json.dumps(tests, indent=1)[:1500])


if __name__ == "__main__":
    main()
