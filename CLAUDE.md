# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Simulation

```bash
# Activate venv first
source venv/bin/activate

# Full run (medium mode, ~2-3 hours on Apple Silicon)
python morphing_glider.py

# Change RUN_MODE on line 157 to control scale:
#   "dev"    — 1 seed, 20K steps/phase, ~5 min (smoke test)
#   "medium" — 3 seeds, 300K steps/phase, ~2-3 hrs (default)
#   "paper"  — 5 seeds, 1M steps/phase, ~12+ hrs (publication quality)
```

There is no test suite, Makefile, or build system. The script auto-installs missing pip dependencies at import time.

## Architecture

**Single monolithic file** (`morphing_glider.py`, ~5300 lines) organized into numbered `# SECTION:` blocks. Key architecture from top to bottom:

1. **Physics layer**: `RealTimeBezierSpar` (morphing wing geometry via cubic Bezier with curvature constraints), `AeroProxy3D` (quasi-steady strip-theory aerodynamics with panel integration), `DomainRandomizer` (16+ physics parameters randomized for sim2real)

2. **Environment**: `MorphingGliderEnv6DOF` (Gymnasium env) — 41D observation, 6D action (right/left wing tip dx/dy/dz offsets). Quaternion-based 6DOF dynamics at DT=0.04s. `RewardComputer` implements a 9-term reward: Gaussian tracking bonus + survival bonus − attitude/rates/control/jerk/power/structural/symmetry/wall penalties.

3. **Wrappers**: `ProgressiveTwistWrapper` (ramps twist_factor across curriculum phases, applies reward shaping), `ResidualHeuristicWrapper` (learned residual on top of heuristic baseline)

4. **Baselines**: `VirtualTendonHeuristicController`, `PIDYawController`, `LQRYawController`, `LinearMPCYawController`, `GainScheduledPIDYawController` — all implement `.predict(obs)` → `(action, state)`

5. **Training**: `train_baseline_sac()` (flat SAC), `train_with_curriculum()` (5-phase curriculum with gating). Both use SB3's SAC with `VecNormalize` and `SubprocVecEnv`. Curriculum phases defined in `PHASES` list near line 4730.

6. **Evaluation**: `evaluate_controller()` runs episodes, `compute_episode_metrics()` extracts per-episode RMS/settling/failure stats, hierarchical bootstrap provides CIs across seeds.

7. **Interpretability**: `MorphingStrategyAnalyzer` (wing asymmetry analysis), `LatentSpaceMRI` (PCA/tSNE of hidden activations), `PolicySensitivityAnalyzer` (finite-difference Jacobian), `KANPolicyNetwork` + `DAggerDistillation` (knowledge distillation), `SymbolicDistiller` (polynomial regression).

## Key Constants and Their Interactions

- **Yaw authority budget**: `NOMINAL_PHYS["d_yaw"]` and aero moment from `AeroProxy3D` determine the physically achievable yaw rate. Steady-state ceiling ≈ max(ΔMz) / d_yaw. If targets in `DEFAULT_YAW_TARGETS` exceed this, the agent cannot succeed.
- **Actuator lag**: `actuator_tau` in `make_env()` and `MorphingGliderEnv6DOF.__init__()` controls first-order tip response (alpha_act = 1 − exp(−DT/tau)).
- **Reward tracking sharpness**: `REWARD_TRACKING_SHARPNESS` controls exp(−k·e²) width. Higher = sharper gradient near zero error.
- **Curriculum stability_weight**: Per-phase values in `PHASES` should decrease monotonically as phases progress (less attitude penalty as the agent gains competence).

## Output Artifacts

- `morphing_glider_models/` — SB3 `.zip` models, `.vecnorm.pkl`, `.meta.json` per algo/seed
- `morphing_glider_figures/` — Publication plots (asymmetry, DAgger, feature importance, latent space)
- `morphing_glider_results.json` — Full config, calibration, validation, paired tests
- `tb_baseline/`, `tb_curriculum/`, `tb_curriculum_residual/` — TensorBoard logs
- `learning_curves.png`, `aero_sanity.png`, `yaw_overlay.png` — Top-level summary plots

## Hardware

Auto-detects Apple Silicon MPS via `torch.backends.mps.is_available()`, falls back to CPU. `VECENV_MODE = "subproc"` with `fork` start method for parallel training environments.
