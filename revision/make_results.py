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
    ax.plot(t, A["yaw_rate"][j], color=figstyle.SLOT["blue"], lw=1.0, label=label("baseline"))
    ax.plot(t, B["yaw_rate"][j], color=figstyle.CLASSICAL_GREYS[1], lw=1.0, label=label(best))
    ax.set_ylabel("yaw rate (rad/s)"); ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=5.6, frameon=False, borderaxespad=0.1); figstyle.panel_label(ax, "a", x=-0.14)
    ax = axs[1]
    ax.plot(t, np.degrees(A["roll"][j]), color=figstyle.SLOT["blue"], lw=1.0, label="roll"); ax.plot(t, np.degrees(A["pitch"][j]), color=figstyle.SLOT["blue"], lw=0.8, ls=":", label="pitch")
    ax.plot(t, np.degrees(B["roll"][j]), color=figstyle.CLASSICAL_GREYS[1], lw=1.0); ax.plot(t, np.degrees(B["pitch"][j]), color=figstyle.CLASSICAL_GREYS[1], lw=0.8, ls=":")
    ax.set_ylabel("attitude (deg)"); ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=5.6, frameon=False, borderaxespad=0.1); figstyle.panel_label(ax, "b", x=-0.14)
    ax = axs[2]
    from interpretability import virtual
    V = virtual(A["action"][j])
    for c, nm, col in ((0, "diff. sweep", figstyle.SLOT["blue"]), (2, "diff. dihedral", figstyle.SLOT["orange"]), (4, "diff. span", figstyle.SLOT["aqua"]), (3, "sym. dihedral", figstyle.SLOT["violet"])):
        ax.plot(t, V[:, c], color=col, lw=0.9, label=nm)
    ax.set_ylabel("command (m)"); ax.set_xlabel("time (s)"); ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=5.4, frameon=False, borderaxespad=0.1, handlelength=1.2, columnspacing=0.8); figstyle.panel_label(ax, "c", x=-0.14)
    fig.tight_layout(h_pad=1.1); figstyle.save(fig, os.path.join(FIG, "fig_timehistory"))


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


ROBUST_LABELS = [("nominal", "Nominal"), ("tau_100", r"Actuator $\tau$ = 100 ms"), ("tau_140", r"Actuator $\tau$ = 140 ms"),
                 ("struct_mode", "Structural mode 2.65 Hz"), ("aero_v1", "Aero model of submission"),
                 ("lift_low", r"Lift slope $\times$0.6"), ("mass_inertia", r"Mass $\times$1.4, inertia $\times$1.6"),
                 ("gusts_x2", r"Gusts $\times$2"), ("noise_x3", r"Sensor noise $\times$3"),
                 ("airframe_v2", "Lumped airframe (v2)"), ("cg_aft", "Static margin 3%")]
ENVELOPE = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]


def robust_rows():
    """Per condition: SAC (5 seeds, hierarchical CI) and each classical controller in eval_robust/."""
    rows = {}
    conds = [c for c, _ in ROBUST_LABELS] + [f"env_{v:.2f}" for v in ENVELOPE]
    cls = sorted({os.path.basename(f)[len("classical_"):].rsplit("_", 1)[0]
                  for f in glob.glob(os.path.join(RB, "classical_*_nominal.json"))})
    for cond in conds:
        per = [load(os.path.join(RB, f"sac_baseline_seed{s}_{cond}.json")) for s in TRAIN_SEEDS]
        per = [d["episodes"] for d in per if d]
        if per:
            rows[("SAC", cond)] = {k: hier_ci([ep_values(e, k) for e in per]) for k in ("rms_yaw_horizon", "failure")}
        for c in cls:
            d = load(os.path.join(RB, f"classical_{c}_{cond}.json"))
            if d: rows[(c, cond)] = {k: boot_ci(ep_values(d["episodes"], k)) for k in ("rms_yaw_horizon", "failure")}
    return rows, cls


def fig_robustness(rows, cls):
    if not rows: return
    names = ["SAC"] + cls; cols = [color("baseline")] + [figstyle.CLASSICAL_GREYS[i] for i in (0, 3)][:len(cls)]
    labs = [label("baseline")] + [label(c) for c in cls]
    fig = plt.figure(figsize=(figstyle.DOUBLE, 3.6)); gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.7, 1.0], hspace=0.55, wspace=0.35)
    axr = fig.add_subplot(gs[:, 0]); axf = fig.add_subplot(gs[:, 1], sharey=axr)
    axe = fig.add_subplot(gs[0, 2]); axe2 = fig.add_subplot(gs[1, 2], sharex=axe)
    conds = [(c, l) for c, l in ROBUST_LABELS if ("SAC", c) in rows]; y = np.arange(len(conds))[::-1]
    off = np.linspace(0.18, -0.18, len(names))
    for j, (n, col) in enumerate(zip(names, cols)):
        for i, (c, _) in enumerate(conds):
            if (n, c) not in rows: continue
            for ax, k, sc in ((axr, "rms_yaw_horizon", 1), (axf, "failure", 100)):
                m, lo, hi = (v * sc for v in rows[(n, c)][k])
                ax.plot([lo, hi], [y[i] + off[j]] * 2, color=col, lw=0.8); ax.plot(m, y[i] + off[j], "o", ms=2.8, color=col)
    axr.set_yticks(y); axr.set_yticklabels([l for _, l in conds]); axr.set_xlabel("RMS@H (rad/s)"); axr.set_xlim(left=0)
    axf.set_xlabel("failure rate (%)"); axf.set_xlim(0, 100); plt.setp(axf.get_yticklabels(), visible=False)
    for ax in (axr, axf): ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    for j, (n, col, lb) in enumerate(zip(names, cols, labs)):
        pts = [(v, rows[(n, f"env_{v:.2f}")]) for v in ENVELOPE if (n, f"env_{v:.2f}") in rows]
        if not pts: continue
        v = np.array([p[0] for p in pts])
        for ax, k, sc in ((axe, "rms_yaw_horizon", 1), (axe2, "failure", 100)):
            m = np.array([p[1][k][0] for p in pts]) * sc; lo = np.array([p[1][k][1] for p in pts]) * sc
            hi = np.array([p[1][k][2] for p in pts]) * sc
            ax.fill_between(v, lo, hi, color=col, alpha=0.15, lw=0); ax.plot(v, m, "-o", color=col, ms=2.6, lw=1.0, label=lb)
    axe.plot(ENVELOPE, ENVELOPE, color=figstyle.MUTED, lw=0.6, ls="--", label="zero tracking")
    axe.set_ylabel("RMS@H (rad/s)"); axe2.set_ylabel("failure (%)"); axe2.set_xlabel(r"reference magnitude $|r_{\mathrm{ref}}|$ (rad/s)")
    axe2.set_ylim(0, 100); axe.legend(loc="upper left", fontsize=5.5); plt.setp(axe.get_xticklabels(), visible=False)
    for ax, l, x in ((axr, "a", -0.62), (axf, "b", -0.08), (axe, "c", -0.22), (axe2, "d", -0.22)): figstyle.panel_label(ax, l, x=x)
    figstyle.save(fig, os.path.join(FIG, "fig_robustness"))


def tuning_table():
    """LaTeX rows: grid optimum, refined optimum, gain, failure at the selection, plateau and bound checks."""
    from refine_classical import BOUNDS
    ct = load(os.path.join(RUNS_DIR, "analysis", "classical_tuning.json")) or {}
    lines, mac = [], {}
    for n in ["Heuristic", "Alloc-ML", "PID", "GS-PID", "LQR", "MPC"]:
        t = ct.get(n)
        if not t or "stage1_best" not in t: continue
        SYM = {"scale": "$s$", "gain": "$g$", "x_range": "$x_r$", "roll_gain": "$k_\\phi$", "k_att": "$g_{\\mathrm{att}}$",
               "bank_max_deg": "$\\phi_{\\max}$", "k_yaw": "$g_{\\mathrm{yaw}}$"}
        UNIT = {"scale": "~m", "x_range": "~m", "bank_max_deg": "$^\\circ$"}
        sel = ", ".join(f"{SYM.get(k, k)} = {v:.3g}{UNIT.get(k, '')}" for k, v in sorted(t["selected"].items()))
        pts = sorted([r["rms_h"] for r in t["grid"] + t["refinement"]])
        best = t["selected_rms_h"]
        plateau = 100 * (np.mean(pts[1:4]) - best) / best if len(pts) >= 4 else float("nan")   # next 3 best points
        fsel = min(t["grid"] + t["refinement"], key=lambda r: r["rms_h"])["failure"]
        at_bound = [k for k, v in t["selected"].items() if k in BOUNDS and (abs(math.log(v / BOUNDS[k][0])) < 0.02
                                                                              or abs(math.log(v / BOUNDS[k][1])) < 0.02)]
        lines.append(f"{label(n)} & {t['stage1_best']['rms_h']:.3f} & {100 * t['stage1_best']['failure']:.0f} & "
                     f"{best:.3f} & {100 * fsel:.0f} & {plateau:.1f} & {', '.join(SYM.get(a, a) for a in at_bound) or '--'} & {sel}\\\\")
        tg = {"Heuristic": "Heur", "Alloc-ML": "Alloc", "PID": "Pid", "GS-PID": "Gspid", "LQR": "Lqr", "MPC": "Mpc"}[n]
        mac[f"Tune{tg}Grid"] = f"{t['stage1_best']['rms_h']:.3f}"; mac[f"Tune{tg}Ref"] = f"{best:.3f}"
        mac[f"Tune{tg}GridFail"] = f"{100 * t['stage1_best']['failure']:.0f}"; mac[f"Tune{tg}RefFail"] = f"{100 * fsel:.0f}"
        mac[f"Tune{tg}Plateau"] = f"{plateau:.1f}"
    os.makedirs(FIG, exist_ok=True)
    open(os.path.join(FIG, "tab_tuning_rows.tex"), "w").write("\n".join(lines) + "\n")
    return mac


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
    def pfm(v):   # p-value in LaTeX math, e.g. 3.4\times10^{-41}
        if v is None or not np.isfinite(v): return "\\textbf{??}"
        if v >= 1e-3: return f"{v:.3f}"
        e = int(math.floor(math.log10(v))); m = v / 10 ** e
        return f"{m:.1f}\\times10^{{{e}}}"
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
        mac[f"RRatioBest{c}"] = fm(t.get("ratio", np.nan), "{:.2f}"); mac[f"RPBest{c}"] = pfm(t.get("p_wilcoxon", np.nan))
        mac[f"RRedBest{c}"] = fm(100 * (1 - t.get("ratio", np.nan)), "{:.0f}"); mac[f"RFactorBest{c}"] = fm(1 / t.get("ratio", np.nan), "{:.1f}")
        for other, o in (("curriculum", "Curr"), ("residual_curriculum", "Resid")):
            t2 = tests.get(f"baseline_vs_{other}|{cond}") or {}
            mac[f"RPBaseVs{o}{c}"] = pfm(t2.get("p_mannwhitney", np.nan)); mac[f"RDBaseVs{o}{c}"] = fm(t2.get("cohen_d", np.nan), "{:.2f}")
    out["macros"] = mac
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "results_summary.json"))
    fig_performance(S); fig_extended(S); fig_timehistory(data); fig_learning()
    rows, cls = robust_rows(); fig_robustness(rows, cls); mac.update(tuning_table())
    rob = {f"{n}|{c}": v for (n, c), v in rows.items()}; out["robustness"] = rob
    for (n, c), v in rows.items():
        if c.startswith("env_"): continue
        key = "".join(w.capitalize() for w in c.replace("x", "X").split("_")); who = "Sac" if n == "SAC" else tag.get(n, n)
        mac[f"Rob{who}{key}Rmsh"] = fm(v["rms_yaw_horizon"][0]); mac[f"Rob{who}{key}Fail"] = fm(100 * v["failure"][0], "{:.0f}")
    try:   # original curriculum selected checkpoints on randomized episodes 0-24 -> check on episodes 25-49 only
        from common import EVAL_SEED_BASE
        for algo, t in (("baseline", "Base"), ("curriculum", "Curr"), ("residual_curriculum", "Resid")):
            seeds = data.get((algo, "randomized"))
            if not seeds: continue
            per = [np.array([e["rms_yaw_horizon"] for e in v if e["seed"] - EVAL_SEED_BASE >= 25]) for v in seeds.values()]
            fl = [np.array([e["failure"] for e in v if e["seed"] - EVAL_SEED_BASE >= 25]) for v in seeds.values()]
            m, lo, hi = hier_ci(per); mac[f"R{t}RandRmshHeld"] = fm(m); mac[f"R{t}RandRmshHeldLo"] = fm(lo); mac[f"R{t}RandRmshHeldHi"] = fm(hi)
            mac[f"R{t}RandFailHeld"] = fm(100 * float(np.mean(np.concatenate(fl))), "{:.0f}")
            out.setdefault("heldout_25_49", {})[algo] = {"rms_h": [m, lo, hi], "failure": float(np.mean(np.concatenate(fl)))}
    except Exception as e:
        print("held-out check skipped:", repr(e))
    out["macros"] = mac; dump_json(out, os.path.join(RUNS_DIR, "analysis", "results_summary.json"))
    for key in sorted(S):
        r = S[key]; print(f"{key:<30} RMS@H {r['rms_yaw_horizon'][0]:.3f} [{r['rms_yaw_horizon'][1]:.3f},{r['rms_yaw_horizon'][2]:.3f}] "
                          f"fail {100*r['failure'][0]:.0f}% settle {r['mean_settle_time'][0]:.2f}s roll {r['rms_roll_deg'][0]:.1f}deg (seeds {r['n_seeds']})")
    print(json.dumps(tests, indent=1)[:1500])


if __name__ == "__main__":
    main()
