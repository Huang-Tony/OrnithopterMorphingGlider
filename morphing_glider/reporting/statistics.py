from typing import Dict, List

import numpy as np

try:
    import scipy.stats as spstats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    spstats = None


def print_statistical_evidence_summary(paired_cache: Dict, power_result: Dict, eval_blocks: List) -> None:
    """Print a consolidated statistical evidence summary.

    Args:
        paired_cache: Dict of paired test results.
        power_result: Output of statistical_power_analysis.
        eval_blocks: Final eval blocks for context.

    Returns:
        None. Prints formatted summary.

    References:
        [COLAS_2019] How Many Seeds? Statistical Power in Deep RL.
    """
    print("\n" + "="*80)
    print("STATISTICAL EVIDENCE SUMMARY")
    print("="*80)
    print(f"\n1. Statistical Power:")
    print(f"   Power (t-test): {power_result.get('power_ttest', np.nan):.3f}")
    print(f"   Power (Wilcoxon): {power_result.get('power_wilcoxon', np.nan):.3f}")
    print(f"   Min detectable d: {power_result.get('min_detectable_d', np.nan):.3f}")
    n_eff = power_result.get('n_effective', power_result.get('n_seeds', 0))
    print(f"   N_seeds={power_result.get('n_seeds',0)}, N_ep/seed={power_result.get('n_episodes_per_seed',0)}, N_effective={n_eff}")
    note = power_result.get('note', '')
    if note:
        print(f"   Note: {note}")

    print(f"\n2a. Controller vs Heuristic (Holm-Bonferroni Corrected):")
    for cond, block in paired_cache.get("controller_vs_heuristic", {}).items():
        print(f"   Condition: {cond}")
        hb = block.get("holm_bonferroni", {})
        for name, res in hb.items():
            rej = "REJECT H0" if res.get("reject", False) else "fail to reject"
            print(f"     {name}: p={res.get('p', np.nan):.6f} -> {rej}")

    print(f"\n2b. Baseline vs Trained (Holm-Bonferroni Corrected, seed-level):")
    for cond, block in paired_cache.get("baseline_vs_trained", {}).items():
        print(f"   Condition: {cond}")
        hb = block.get("holm_bonferroni", {})
        for name, res in hb.items():
            rej = "REJECT H0" if res.get("reject", False) else "fail to reject"
            print(f"     {name}: p={res.get('p', np.nan):.6f} -> {rej}")
        for key in ["baseline_vs_curriculum", "baseline_vs_residual_curriculum"]:
            r = block.get(key, {})
            d = r.get("cohen_d", np.nan)
            d_paired = r.get("cohen_d_paired", np.nan)
            md = r.get("mean_diff", np.nan)
            p = r.get("p_wilcoxon", r.get("p_ttest", np.nan))
            p_mw = r.get("p_mannwhitney", np.nan)
            if np.isfinite(d):
                direction = "worse" if md > 0 else "better"
                size = "negligible" if abs(d)<0.2 else ("small" if abs(d)<0.5 else ("medium" if abs(d)<0.8 else "large"))
                parts = [f"{size} effect (d={d:+.3f}, d_paired={d_paired:+.3f})"]
                if np.isfinite(p):
                    parts.append(f"p_wilc={p:.4f}")
                if np.isfinite(p_mw):
                    parts.append(f"p_mw={p_mw:.4f}")
                parts.append(f"{direction} than baseline")
                print(f"     {key} ({cond}): {', '.join(parts)}")

    print(f"\n3. Plain-language conclusions:")
    controller_names = ["PID_vs_Heuristic", "LQR_vs_Heuristic",
                        "MPC_vs_Heuristic", "GS-PID_vs_Heuristic",
                        "baseline_vs_Heuristic", "curriculum_vs_Heuristic",
                        "residual_curriculum_vs_Heuristic"]
    for cond, block in paired_cache.get("controller_vs_heuristic", {}).items():
        for name in controller_names:
            r = block.get(name, {})
            d = r.get("cohen_d", np.nan); md = r.get("mean_diff", np.nan)
            p = r.get("p_wilcoxon", r.get("p_ttest", np.nan))
            if np.isfinite(d):
                direction = "worse" if md > 0 else "better"
                size = "negligible" if abs(d)<0.2 else ("small" if abs(d)<0.5 else ("medium" if abs(d)<0.8 else "large"))
                sig = f", p={p:.4f}" if np.isfinite(p) else ""
                print(f"   {name} ({cond}): {size} effect (d={d:+.3f}){sig}, {direction} than heuristic")


def print_metric_correlations(eval_blocks: List) -> None:
    """Print Pearson/Spearman correlations between key metric pairs.

    Args:
        eval_blocks: List of (algo, cond, block) tuples from final eval.
    """
    METRIC_PAIRS = [
        ("rms_yaw_horizon", "failure"),
        ("rms_yaw_steady", "mean_settle_time"),
        ("mean_action_norm", "mean_power_loss"),
        ("rms_yaw_horizon", "mean_action_norm"),
    ]

    print("\n" + "="*80)
    print("METRIC CORRELATION ANALYSIS")
    print("="*80)

    for algo, cond, block in eval_blocks:
        raw = block.get("raw_metrics", [])
        seed_eps = block.get("seed_episodes", {})
        if seed_eps:
            raw = []
            for seed_mets in seed_eps.values():
                raw.extend(seed_mets)
        if not raw or len(raw) < 5:
            continue

        has_correlation = False
        for m1_key, m2_key in METRIC_PAIRS:
            v1 = np.array([float(m.get(m1_key, np.nan)) for m in raw])
            v2 = np.array([float(m.get(m2_key, np.nan)) for m in raw])
            mask = np.isfinite(v1) & np.isfinite(v2)
            if mask.sum() < 5:
                continue
            v1f = v1[mask]; v2f = v2[mask]
            if np.std(v1f) < 1e-12 or np.std(v2f) < 1e-12:
                continue

            if not has_correlation:
                print(f"\n  {algo} ({cond}):")
                has_correlation = True

            if _HAS_SCIPY:
                r_p, p_p = spstats.pearsonr(v1f, v2f)
                r_s, p_s = spstats.spearmanr(v1f, v2f)
                print(f"    {m1_key} vs {m2_key}: Pearson r={r_p:.3f} (p={p_p:.4f}), Spearman rho={r_s:.3f} (p={p_s:.4f})")
            else:
                r_p = float(np.corrcoef(v1f, v2f)[0, 1])
                print(f"    {m1_key} vs {m2_key}: Pearson r={r_p:.3f}")
