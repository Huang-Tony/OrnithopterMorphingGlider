def generate_methods_comment_block() -> str:
    """Generate a formatted Methods section as a comment block.

    Returns:
        String containing the formatted methods text.

    References:
        [NMI_2024] Nature Machine Intelligence author guidelines.
    """
    text = """
# ================================================================
# METHODS (for manuscript supplementary)
# ================================================================
#
# Reward Function (restructured for positive survival incentive):
#   r = survival_bonus
#     + w_track * exp(-sharpness * e_r^2)  [positive tracking]
#     - w_att * (roll^2 + pitch^2)/ref     [attitude penalty]
#     - w_rates * (wp^2 + wq^2)/ref        [rate penalty]
#     - w_ctrl * ||u||^2 - w_jerk * ||Du||^2 - w_power * P_loss
#     - w_struct * E_struct - w_zsym * asym_z^2
#     - w_wall * soft_wall(roll, pitch)     [exponential near limits]
#   Terminal: base * lambda(s) + remaining * survival_bonus
#   where lambda(s) = 1 + 3*(1 - s), s = step/max_steps.
#
# Statistical Methods:
#   - All CIs: BCa bootstrap (N=6000 resamples for paper mode)
#   - Multi-seed: hierarchical bootstrap over training seeds (outer)
#     and eval episodes (inner)
#   - Paired tests: Wilcoxon signed-rank + paired t-test
#   - Multiple comparisons: Holm-Bonferroni correction at alpha=0.05
#   - Effect sizes: Cohen's d for all paired comparisons
#
# Aero Proxy Validation:
#   - AeroProxy3D compared against VortexLatticeReference (horseshoe VLM)
#   - Pearson r, RMSE, max relative deviation reported
#   - Mark: [HW_VALIDATION_REQUIRED: compare against wind tunnel at Re~1e5]
#
# Spar Proxy Validation:
#   - Bezier curvature energy compared against Euler-Bernoulli bending energy
#   - Mark: [CALIBRATION_REQUIRED: measure actual spar EI from 3-point bend test]
#
# Interpretability Methods:
#   - Machine Teaching: Linear regression extracts slope from
#     (yaw_target -> wing_asymmetry) mapping.  Coefficient injected into
#     VirtualTendonHeuristicController.z_range and GainScheduledPID.
#   - Latent Space MRI: PCA/t-SNE projection of actor latent_pi
#     activations, color-coded by flight regime.
#   - KAN: Two-layer Kolmogorov-Arnold Network (B-spline basis)
#     distilled via DAgger from SAC expert.  Symbolic polynomial
#     extraction via least-squares fit on learned univariate functions.
#   - Symbolic Distillation: Direct polynomial regression from
#     key obs features to actions.  Reports per-action R-squared.
#
# Limitations:
#   1. Surrogate aerodynamic model not validated against CFD or experiment.
#   2. Structural energy proxy does not represent true FEM bending energy.
#   3. Domain randomization bounds chosen heuristically, not from measured uncertainty.
#   4. Training on single machine limits statistical power to N<=5 seeds.
#   5. No real-flight validation; all results are simulation-only.
#   6. KAN symbolic extraction is approximate (polynomial fit to B-splines).
#   7. DAgger distillation fidelity depends on expert state coverage.
"""
    return text
