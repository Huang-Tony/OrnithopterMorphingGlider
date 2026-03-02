from typing import Dict, List

import numpy as np


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
    print(f"   N_seeds={power_result.get('n_seeds',0)}, N_ep/seed={power_result.get('n_episodes_per_seed',0)}")

    print(f"\n2. Holm-Bonferroni Corrected Comparisons:")
    for cond, block in paired_cache.get("controller_vs_heuristic", {}).items():
        print(f"   Condition: {cond}")
        hb = block.get("holm_bonferroni", {})
        for name, res in hb.items():
            rej = "REJECT H0" if res.get("reject", False) else "fail to reject"
            print(f"     {name}: p={res.get('p', np.nan):.6f} -> {rej}")

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
