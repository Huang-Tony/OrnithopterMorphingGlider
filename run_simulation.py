#!/usr/bin/env python3
"""Main entry point for the Morphing Glider simulation.

Replaces the monolithic morphing_glider.py __main__ block.
Run modes: dev / medium / paper (set via morphing_glider.config.RUN_MODE).
"""

import gc
import json
import os
import time
import warnings
import multiprocessing
from typing import Dict, Any, List

import numpy as np

from morphing_glider.config import (
    RUN_MODE, FAST_DEV_RUN, MEDIUM_RUN, PAPER_RUN,
    GLOBAL_SEED, DT, DEVICE, DEFAULT_YAW_TARGETS,
    N_TRAIN_SEEDS, TOTAL_TRAIN_STEPS, EVAL_EPISODES_PER_SEED,
    HYPERPARAMETER_REGISTRY,
    RUN_AERO_CALIBRATION, RUN_AERO_SANITY_SWEEP,
    RUN_TRAIN_BASELINE, RUN_TRAIN_CURRICULUM, RUN_TRAIN_RESIDUAL_CURRICULUM,
    RUN_FINAL_EVAL, RUN_ABLATION_SUITE, RUN_DEMO_OVERLAY,
    REWARD_W_TRACK, REWARD_W_STRUCT,
    seed_everything,
)
from morphing_glider.environment.observation import OBS_DIM
from morphing_glider.environment.wrappers import (
    ProgressiveTwistWrapper, mild_curriculum_reward_shaper,
)
from morphing_glider.physics.domain_randomizer import NOMINAL_PHYS
from morphing_glider.controllers.zero import ZeroController
from morphing_glider.controllers.heuristic import VirtualTendonHeuristicController
from morphing_glider.controllers.pid import PIDYawController, GainScheduledPIDYawController
from morphing_glider.controllers.lqr import LQRYawController
from morphing_glider.controllers.mpc import LinearMPCYawController
from morphing_glider.controllers.sb3_controller import SB3Controller
from morphing_glider.training.curriculum import PhaseSpec, TrainRunResult
from morphing_glider.utils.numeric import paired_tests, holm_bonferroni


def main():
    multiprocessing.freeze_support()
    os.makedirs("morphing_glider_models", exist_ok=True)
    os.makedirs("morphing_glider_figures", exist_ok=True)

    seed_everything(GLOBAL_SEED)

    # ---- Runtime sizing ----
    if FAST_DEV_RUN:
        N_ENVS = 2; MAX_STEPS_EP = 180; BASELINE_STEPS = TOTAL_TRAIN_STEPS
        CURR_EVAL_EVERY = 8_000; CURR_EVAL_EPS = 5; FINAL_EVAL_EPS = EVAL_EPISODES_PER_SEED
    elif MEDIUM_RUN:
        N_ENVS = 6; MAX_STEPS_EP = 200; BASELINE_STEPS = TOTAL_TRAIN_STEPS
        CURR_EVAL_EVERY = 16_000; CURR_EVAL_EPS = 20; FINAL_EVAL_EPS = EVAL_EPISODES_PER_SEED
    else:
        N_ENVS = 10; MAX_STEPS_EP = 200; BASELINE_STEPS = TOTAL_TRAIN_STEPS
        CURR_EVAL_EVERY = 50_000; CURR_EVAL_EPS = 25; FINAL_EVAL_EPS = EVAL_EPISODES_PER_SEED

    EVAL_SEED_BASE = int(GLOBAL_SEED + 10000)

    if PAPER_RUN and N_TRAIN_SEEDS < 3:
        warnings.warn("Paper run with fewer than 3 seeds.", RuntimeWarning)

    # ---- Curriculum phases ----
    PHASES = [
        PhaseSpec(name="basic_yaw", twist_factor=0.0, rand_scale=0.0,
                  max_timesteps=TOTAL_TRAIN_STEPS,
                  ramp_steps=1800 if PAPER_RUN else (900 if MEDIUM_RUN else 500),
                  reward_shaper=mild_curriculum_reward_shaper,
                  residual_limit=np.array([0.05, 0.025, 0.03, 0.05, 0.025, 0.03], dtype=float),
                  learning_rate=3e-4, target_rms=0.52, min_steps_before_gate=20_000,
                  roll_pitch_limit_deg=90.0, coupling_scale=0.10, stability_weight=0.08),
        PhaseSpec(name="partial_twist", twist_factor=0.3, rand_scale=0.3,
                  max_timesteps=TOTAL_TRAIN_STEPS,
                  ramp_steps=3000 if PAPER_RUN else (1800 if MEDIUM_RUN else 1000),
                  reward_shaper=mild_curriculum_reward_shaper,
                  residual_limit=np.array([0.05, 0.025, 0.03, 0.05, 0.025, 0.03], dtype=float),
                  learning_rate=3e-4, target_rms=0.38, min_steps_before_gate=30_000,
                  roll_pitch_limit_deg=88.0, coupling_scale=0.25, stability_weight=0.06),
        PhaseSpec(name="moderate_twist", twist_factor=0.6, rand_scale=0.6,
                  max_timesteps=TOTAL_TRAIN_STEPS,
                  ramp_steps=3500 if PAPER_RUN else (2000 if MEDIUM_RUN else 1200),
                  reward_shaper=mild_curriculum_reward_shaper,
                  residual_limit=np.array([0.10, 0.05, 0.06, 0.10, 0.05, 0.06], dtype=float),
                  learning_rate=3e-4, target_rms=0.28, min_steps_before_gate=40_000,
                  roll_pitch_limit_deg=82.0, coupling_scale=0.50, stability_weight=0.05),
        PhaseSpec(name="full_twist", twist_factor=1.0, rand_scale=1.0,
                  max_timesteps=TOTAL_TRAIN_STEPS,
                  ramp_steps=4000 if PAPER_RUN else (2500 if MEDIUM_RUN else 1500),
                  reward_shaper=mild_curriculum_reward_shaper,
                  residual_limit=np.array([0.10, 0.05, 0.06, 0.10, 0.05, 0.06], dtype=float),
                  learning_rate=2e-4, target_rms=0.20, min_steps_before_gate=50_000,
                  roll_pitch_limit_deg=70.0, coupling_scale=0.90, stability_weight=0.04),
        PhaseSpec(name="raw_finetune", twist_factor=1.0, rand_scale=1.0,
                  max_timesteps=TOTAL_TRAIN_STEPS, ramp_steps=0, reward_shaper=mild_curriculum_reward_shaper,
                  residual_limit=np.array([0.20, 0.10, 0.12, 0.20, 0.10, 0.12], dtype=float),
                  learning_rate=1e-4, target_rms=None, min_steps_before_gate=60_000,
                  roll_pitch_limit_deg=65.0, coupling_scale=1.00, stability_weight=0.035),
    ]

    # ---- Controllers ----
    heuristic = VirtualTendonHeuristicController(
        yaw_rate_max=max(abs(v) for v in DEFAULT_YAW_TARGETS))
    zero = ZeroController()

    # ---- Helper: train and save ----
    def train_and_save_one(algo, seed):
        from morphing_glider.training.baseline import train_baseline_sac
        from morphing_glider.training.curriculum import train_with_curriculum, summarize_curriculum_progression
        from morphing_glider.training.infrastructure import save_model_and_vecnorm, save_training_checkpoint

        out_dir = "morphing_glider_models"
        if algo == "baseline":
            model, vecnorm, logs = train_baseline_sac(
                total_timesteps=BASELINE_STEPS, seed=seed, n_envs=N_ENVS,
                max_steps=MAX_STEPS_EP, eval_every_steps=CURR_EVAL_EVERY,
                eval_episodes=CURR_EVAL_EPS, eval_seed_base=EVAL_SEED_BASE,
                eval_domain_rand_scale=1.0)
        elif algo == "curriculum":
            model, vecnorm, logs = train_with_curriculum(
                phases=PHASES, seed=seed, n_envs=N_ENVS,
                max_steps=MAX_STEPS_EP, eval_every_steps=CURR_EVAL_EVERY,
                eval_episodes=CURR_EVAL_EPS, eval_seed_base=EVAL_SEED_BASE,
                use_residual=False)
        elif algo == "residual_curriculum":
            model, vecnorm, logs = train_with_curriculum(
                phases=PHASES, seed=seed, n_envs=N_ENVS,
                max_steps=MAX_STEPS_EP, eval_every_steps=CURR_EVAL_EVERY,
                eval_episodes=CURR_EVAL_EPS, eval_seed_base=EVAL_SEED_BASE,
                use_residual=True)
        else:
            raise ValueError(algo)
        name = f"{algo}_seed{seed}"
        model_path, vecnorm_path = save_model_and_vecnorm(model, vecnorm, out_dir=out_dir, name=name)
        meta = {
            "algo": algo, "seed": seed,
            "mean_rmsh": float(logs.get("evaltrace", [{}])[-1].get("mean_rmsh", np.nan))
            if logs.get("evaltrace") else float("nan"),
        }
        save_training_checkpoint(model, model_path, meta)
        summarize_curriculum_progression(logs)
        try:
            model.env.close()
        except Exception:
            pass
        del model; gc.collect()
        return TrainRunResult(algo_name=algo, train_seed=int(seed), model_path=model_path,
                              vecnorm_path=vecnorm_path, train_logs=logs)

    # ---- Results containers ----
    aero_calib_out: Dict[str, Any] = {}
    aero_validation_out: Dict[str, Any] = {}
    spar_validation_out: Dict[str, Any] = {}
    train_runs: List[TrainRunResult] = []
    final_eval_blocks: list = []
    paired_test_cache: Dict[str, Any] = {}
    ablation_results: Dict[str, Any] = {}
    robustness_results: Dict[str, Any] = {}
    repro_report: Dict[str, Any] = {}

    train_seeds = [GLOBAL_SEED + 100 * s for s in range(int(N_TRAIN_SEEDS))]
    FINAL_EVAL_RPL = float(PHASES[-1].roll_pitch_limit_deg)
    FINAL_EVAL_COUP = 1.0
    FINAL_EVAL_SW = 0.03

    # ---- Calibration ----
    if RUN_AERO_CALIBRATION:
        from morphing_glider.calibration import aero_calibration
        from morphing_glider.physics.validators import validate_aero_proxy
        aero_calib_out = aero_calibration()
        aero_validation_out = validate_aero_proxy(dict(NOMINAL_PHYS), n_alpha=12)

    if RUN_AERO_SANITY_SWEEP:
        from morphing_glider.calibration import aero_sanity_sweep
        from morphing_glider.physics.validators import validate_spar_proxy
        aero_sanity_sweep()
        spar_validation_out = validate_spar_proxy(n_deflections=8)

    # ---- Training ----
    if RUN_TRAIN_BASELINE or RUN_TRAIN_CURRICULUM or RUN_TRAIN_RESIDUAL_CURRICULUM:
        print(f"\n{'#' * 80}\nTRAINING | mode={RUN_MODE} seeds={train_seeds} envs={N_ENVS}\n{'#' * 80}")

    if RUN_TRAIN_BASELINE:
        for sd in train_seeds:
            train_runs.append(train_and_save_one("baseline", sd))
    if RUN_TRAIN_CURRICULUM:
        for sd in train_seeds:
            train_runs.append(train_and_save_one("curriculum", sd + 10))
    if RUN_TRAIN_RESIDUAL_CURRICULUM:
        for sd in train_seeds:
            train_runs.append(train_and_save_one("residual_curriculum", sd + 20))

    # ---- Checkpoint verification ----
    if train_runs:
        from morphing_glider.training.infrastructure import verify_checkpoint_reproducibility, make_env
        best_rr = train_runs[0]
        print("\n[CHECKPOINT VERIFICATION]")
        passed = verify_checkpoint_reproducibility(
            best_rr.model_path,
            env_factory=lambda: make_env(seed=0, domain_rand_scale=0.0, max_steps=200, for_eval=True),
            n_episodes=5)
        print(f"REPRODUCIBILITY: {'PASS' if passed else 'FAIL'}")

    # ---- Learning curves ----
    if train_runs:
        from morphing_glider.utils.visualization import plot_learning_curves
        algo_to_logs = {}
        for rr in train_runs:
            if rr.algo_name not in algo_to_logs:
                algo_to_logs[rr.algo_name] = rr.train_logs
        if algo_to_logs:
            plot_learning_curves(algo_to_logs, title=f"Learning curves ({RUN_MODE})")

    # ---- Final evaluation ----
    if RUN_FINAL_EVAL:
        from morphing_glider.evaluation.evaluate import (
            summarize_controller_over_episodes_bca,
            summarize_trained_algo_hierarchical,
            print_final_eval_table,
        )
        from morphing_glider.evaluation.metrics import summarize_metrics
        from morphing_glider.utils.numeric import statistical_power_analysis
        from morphing_glider.reporting.statistics import print_statistical_evidence_summary

        print(f"\n{'#' * 80}\nFINAL EVALUATION\n{'#' * 80}")

        K_MZ_EST = 2.128
        try:
            if isinstance(aero_calib_out, dict):
                mz = float(aero_calib_out.get("Mz_total", np.nan))
                dx = float(aero_calib_out.get("dx_ref", np.nan))
                if np.isfinite(mz) and np.isfinite(dx) and abs(dx) > 1e-6:
                    K_MZ_EST = abs(mz) / abs(dx)
        except Exception:
            pass

        pid = PIDYawController(dt=DT, action_scale=0.15)
        try:
            pid.auto_tune_from_aero(Izz=float(NOMINAL_PHYS["Izz"]), K_mz_per_dx=float(K_MZ_EST))
        except Exception:
            pass

        lqr = LQRYawController(Izz=float(NOMINAL_PHYS["Izz"]), K_mz_per_dx=float(K_MZ_EST),
                                Q=1.0, R=0.1, dt=DT, action_scale=0.15)
        mpc = LinearMPCYawController(Izz=float(NOMINAL_PHYS["Izz"]), K_mz_per_dx=float(K_MZ_EST),
                                     d_yaw=float(NOMINAL_PHYS["d_yaw"]), dt=DT, action_scale=0.15)
        gs_pid = GainScheduledPIDYawController(dt=DT, action_scale=0.15)
        try:
            gs_pid.auto_tune_from_aero(Izz=float(NOMINAL_PHYS["Izz"]), K_mz_per_dx=float(K_MZ_EST))
        except Exception:
            pass

        baseline_controllers = [
            ("Heuristic", heuristic), ("PID", pid), ("LQR", lqr),
            ("MPC", mpc), ("GS-PID", gs_pid), ("Zero", zero),
        ]

        for cname, ctrl in baseline_controllers:
            for ds, tag in [(0.0, "nominal"), (1.0, "randomized")]:
                result = summarize_controller_over_episodes_bca(
                    ctrl, label=cname, domain_scale=ds, max_steps=MAX_STEPS_EP,
                    eval_episodes=FINAL_EVAL_EPS, eval_seed_base=EVAL_SEED_BASE,
                    roll_pitch_limit_deg=FINAL_EVAL_RPL, coupling_scale=FINAL_EVAL_COUP,
                    stability_weight=FINAL_EVAL_SW, ci=95.0, return_raw_metrics=True)
                final_eval_blocks.append((cname, tag, result))

        # Paired tests
        def _rmsh_list(block):
            raw = block.get("raw_metrics", [])
            if raw:
                return np.array([float(m.get("rms_yaw_horizon", np.nan)) for m in raw])
            seed_eps = block.get("seed_episodes", {})
            if seed_eps:
                vals = []
                for seed_mets in seed_eps.values():
                    for m in seed_mets:
                        vals.append(float(m.get("rms_yaw_horizon", np.nan)))
                return np.array(vals)
            return np.array([])

        paired_test_cache["controller_vs_heuristic"] = {}
        for cond in ["nominal", "randomized"]:
            heur_block = next((b for a, c, b in final_eval_blocks if a == "Heuristic" and c == cond), None)
            if heur_block is None:
                continue
            x = _rmsh_list(heur_block)
            if x.size < 2:
                continue
            pvals = {}; test_details = {}
            for cname, ctag, block in final_eval_blocks:
                if cname == "Heuristic" or ctag != cond:
                    continue
                y = _rmsh_list(block)
                n_pair = min(x.size, y.size)
                if n_pair < 2:
                    continue
                res = paired_tests(x[:n_pair], y[:n_pair])
                p = res.get("p_wilcoxon", np.nan)
                if not np.isfinite(p):
                    p = res.get("p_ttest", np.nan)
                key = f"{cname}_vs_Heuristic"
                pvals[key] = p; test_details[key] = res
            corr = holm_bonferroni(pvals, alpha=0.05)
            paired_test_cache["controller_vs_heuristic"][cond] = {"holm_bonferroni": corr, **test_details}

        # Trained model evaluation
        for ds, tag in [(0.0, "nominal"), (1.0, "randomized")]:
            for algo in ["baseline", "curriculum", "residual_curriculum"]:
                res_lim = PHASES[-1].residual_limit if algo == "residual_curriculum" else None
                summ = summarize_trained_algo_hierarchical(
                    train_runs, algo_name=algo, domain_scale=float(ds), max_steps=MAX_STEPS_EP,
                    eval_episodes=FINAL_EVAL_EPS, eval_seed_base=EVAL_SEED_BASE,
                    roll_pitch_limit_deg=FINAL_EVAL_RPL, coupling_scale=FINAL_EVAL_COUP,
                    stability_weight=FINAL_EVAL_SW, residual_limit=res_lim, ci=95.0)
                final_eval_blocks.append((algo, tag, summ))

        print_final_eval_table(final_eval_blocks, ci=95.0)

        power_result = statistical_power_analysis(
            effect_size=0.5, alpha=0.05,
            n_seeds=N_TRAIN_SEEDS, n_episodes_per_seed=EVAL_EPISODES_PER_SEED)
        print(f"\n[POWER ANALYSIS] {power_result}")
        print_statistical_evidence_summary(paired_test_cache, power_result, final_eval_blocks)

        # ---- Interpretability ----
        _run_interpretability(train_runs, heuristic, pid, gs_pid, MAX_STEPS_EP, FINAL_EVAL_EPS,
                              EVAL_SEED_BASE, FINAL_EVAL_RPL, FINAL_EVAL_COUP, FINAL_EVAL_SW)

    # ---- Ablation ----
    if RUN_ABLATION_SUITE:
        from morphing_glider.evaluation.evaluate import summarize_controller_over_episodes_bca
        from morphing_glider.utils.visualization import plot_ablation_summary

        print(f"\n{'#' * 80}\nABLATION SUITE\n{'#' * 80}")
        ablation_eval_eps = max(5, EVAL_EPISODES_PER_SEED // 5)
        ablation_conditions = {
            "full_model": {"desc": "Full model (reference)", "phases": PHASES},
            "no_domain_rand": {"desc": "domain_rand_scale=0", "dr_scale": 0.0},
        }
        for cond_name, cond_cfg in ablation_conditions.items():
            print(f"\n--- Ablation: {cond_name} ({cond_cfg.get('desc', '')}) ---")
            result = summarize_controller_over_episodes_bca(
                heuristic, label=f"ablation_{cond_name}",
                domain_scale=cond_cfg.get("dr_scale", 1.0), max_steps=MAX_STEPS_EP,
                eval_episodes=ablation_eval_eps, eval_seed_base=EVAL_SEED_BASE + 5000,
                roll_pitch_limit_deg=FINAL_EVAL_RPL, coupling_scale=FINAL_EVAL_COUP,
                stability_weight=cond_cfg.get("stab_w", FINAL_EVAL_SW), ci=95.0)
            ablation_results[cond_name] = result
        plot_ablation_summary(ablation_results)

    # ---- Demo overlay ----
    if RUN_DEMO_OVERLAY:
        from morphing_glider.training.infrastructure import make_env
        from morphing_glider.evaluation.metrics import run_episode
        from morphing_glider.utils.visualization import plot_yaw_overlay

        demo_seed = int(GLOBAL_SEED + 123)
        env_demo = make_env(seed=demo_seed, domain_rand_scale=0.0, max_steps=MAX_STEPS_EP,
                            for_eval=True, roll_pitch_limit_deg=FINAL_EVAL_RPL,
                            coupling_scale=FINAL_EVAL_COUP, stability_weight=FINAL_EVAL_SW)
        env_demo = ProgressiveTwistWrapper(env_demo, phase={"name": "demo"}, twist_factor=1.0, reward_shaper=None)
        histories = [run_episode(env_demo, heuristic, deterministic=True, seed=demo_seed)]
        labels = ["Heuristic"]
        plot_yaw_overlay(histories, labels, title="Yaw tracking overlay (nominal)")

    # ---- Reproducibility report ----
    from morphing_glider.reporting.reproducibility import ReproducibilityReport
    from morphing_glider.reporting.methods import generate_methods_comment_block

    repro_report = ReproducibilityReport.save_and_print()

    # ---- Save results JSON ----
    all_results = {
        "config": dict(HYPERPARAMETER_REGISTRY),
        "NOMINAL_PHYS": dict(NOMINAL_PHYS),
        "aero_calibration": aero_calib_out,
        "aero_validation": aero_validation_out,
        "spar_validation": spar_validation_out,
        "train_runs": [{"algo": rr.algo_name, "seed": rr.train_seed, "model_path": rr.model_path}
                       for rr in train_runs],
        "paired_tests": paired_test_cache,
        "robustness": robustness_results,
        "ablation": {k: {"summaries": v.get("summaries", {})} for k, v in ablation_results.items()},
        "reproducibility": repro_report,
        "interpretability": {
            "device": str(DEVICE),
            "modules": ["MachineTeacher", "LatentSpaceMRI", "KANPolicyNetwork",
                        "SymbolicDistiller", "DAggerDistillation"],
        },
    }
    try:
        with open("morphing_glider_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        print("\n[RESULTS] Saved JSON to morphing_glider_results.json")
    except Exception as e:
        print(f"[RESULTS] JSON save failed: {e!r}")

    print(generate_methods_comment_block())
    print("\nDone.")


def _run_interpretability(train_runs, heuristic, pid, gs_pid,
                          MAX_STEPS_EP, FINAL_EVAL_EPS, EVAL_SEED_BASE,
                          FINAL_EVAL_RPL, FINAL_EVAL_COUP, FINAL_EVAL_SW):
    """Run interpretability analysis pipeline on best trained model."""
    import torch
    from stable_baselines3 import SAC
    from morphing_glider.config import DEVICE, DEFAULT_YAW_TARGETS, FAST_DEV_RUN, MEDIUM_RUN, DT
    from morphing_glider.environment.observation import OBS_DIM
    from morphing_glider.training.infrastructure import load_vecnorm_for_eval
    from morphing_glider.controllers.sb3_controller import SB3Controller
    from morphing_glider.evaluation.evaluate import summarize_controller_over_episodes_bca

    print(f"\n{'#' * 80}")
    print("INTERPRETABILITY ANALYSIS")
    print(f"{'#' * 80}")

    best_model_path = None; best_vecnorm_path = None; best_algo = None
    for rr in train_runs:
        if rr.algo_name == "baseline":
            best_model_path = rr.model_path; best_vecnorm_path = rr.vecnorm_path
            best_algo = rr.algo_name; break
    if best_model_path is None and train_runs:
        rr = train_runs[0]
        best_model_path = rr.model_path; best_vecnorm_path = rr.vecnorm_path; best_algo = rr.algo_name

    if best_model_path is None or not os.path.exists(best_model_path):
        print("[INTERP] No trained model available for interpretability analysis")
        return

    print(f"\n[INTERP] Using model: {best_algo} from {best_model_path}")
    interp_model = SAC.load(best_model_path, device=DEVICE)
    interp_vecnorm = load_vecnorm_for_eval(best_vecnorm_path, max_steps=MAX_STEPS_EP)
    interp_ctrl = SB3Controller(
        interp_model,
        obs_rms=(interp_vecnorm.obs_rms if interp_vecnorm else None),
        clip_obs=(interp_vecnorm.clip_obs if interp_vecnorm else 10.0))

    # 1. Machine Teaching
    try:
        from morphing_glider.interpretability.strategy_analyzer import MorphingStrategyAnalyzer
        from morphing_glider.interpretability.machine_teaching import MachineTeacher

        print(f"\n{'=' * 60}\n1. MACHINE TEACHING\n{'=' * 60}")
        shapes = MorphingStrategyAnalyzer.collect_steady_state_shapes(
            interp_ctrl, env_factory=None, yaw_targets=DEFAULT_YAW_TARGETS,
            n_episodes=3 if FAST_DEV_RUN else 5)
        MorphingStrategyAnalyzer.plot_asymmetry_curve(shapes, save_path="morphing_glider_figures/asymmetry_curve.png")
        learned_coeff = MachineTeacher.extract_learned_coefficient(shapes)
        if np.isfinite(learned_coeff.get("slope", np.nan)):
            MachineTeacher.inject_into_heuristic(heuristic, learned_coeff)
            MachineTeacher.inject_into_gain_scheduled_pid(gs_pid, learned_coeff)
            ai_pid = MachineTeacher.create_ai_enhanced_pid(learned_coeff, Kp=pid.Kp, dt=DT, action_scale=0.15)
            ai_pid_result = summarize_controller_over_episodes_bca(
                ai_pid, label="AI-PID", domain_scale=1.0, max_steps=MAX_STEPS_EP,
                eval_episodes=min(10, FINAL_EVAL_EPS), eval_seed_base=EVAL_SEED_BASE + 7000,
                roll_pitch_limit_deg=FINAL_EVAL_RPL, coupling_scale=FINAL_EVAL_COUP,
                stability_weight=FINAL_EVAL_SW, ci=95.0)
            print(f"  AI-PID RMS@H: {ai_pid_result['summaries'].get('rms_yaw_horizon', {}).get('mean', np.nan):.4f}")
    except Exception as e:
        print(f"  [MACHINE TEACHING] Failed: {e!r}")

    # 2. Latent Space MRI
    try:
        from morphing_glider.interpretability.latent_space import LatentSpaceMRI
        print(f"\n{'=' * 60}\n2. LATENT SPACE MRI\n{'=' * 60}")
        latents, mri_metadata = LatentSpaceMRI.collect_latents_from_policy(
            interp_model, n_episodes=3 if FAST_DEV_RUN else 5, max_steps=MAX_STEPS_EP,
            obs_rms=(interp_vecnorm.obs_rms if interp_vecnorm else None))
        if latents.size > 0 and latents.shape[0] > 10:
            LatentSpaceMRI.visualize(latents, mri_metadata, method="pca",
                                     save_path="morphing_glider_figures/latent_space_mri_pca.png")
            if latents.shape[0] > 60:
                LatentSpaceMRI.visualize(latents, mri_metadata, method="tsne",
                                         save_path="morphing_glider_figures/latent_space_mri_tsne.png")
    except Exception as e:
        print(f"  [LatentMRI] Failed: {e!r}")

    # 3. Policy sensitivity
    try:
        from morphing_glider.interpretability.sensitivity import PolicySensitivityAnalyzer
        print(f"\n{'=' * 60}\n3. POLICY SENSITIVITY\n{'=' * 60}")
        importance = PolicySensitivityAnalyzer.feature_importance(
            interp_ctrl, eval_episodes=[], n_samples=20 if FAST_DEV_RUN else 40)
        PolicySensitivityAnalyzer.plot_feature_importance(
            importance, save_path="morphing_glider_figures/feature_importance.png")
        for i, (fn, fv) in enumerate(list(importance.items())[:5]):
            print(f"    {i + 1}. {fn}: {fv:.6f}")
    except Exception as e:
        print(f"  [FeatureImportance] Failed: {e!r}")

    # 4. Symbolic distillation
    try:
        from morphing_glider.interpretability.symbolic import SymbolicDistiller
        print(f"\n{'=' * 60}\n4. SYMBOLIC DISTILLATION\n{'=' * 60}")
        distiller = SymbolicDistiller(polynomial_degree=3,
                                      key_features=["omega_r", "yaw_ref", "sin_roll", "cos_roll", "speed"])
        obs_data, act_data = distiller.collect_expert_data(
            interp_ctrl, n_episodes=5 if FAST_DEV_RUN else 15, max_steps=MAX_STEPS_EP)
        distill_result = distiller.fit(obs_data, act_data)
        sym_result = summarize_controller_over_episodes_bca(
            distiller, label="Symbolic", domain_scale=0.5, max_steps=MAX_STEPS_EP,
            eval_episodes=min(10, FINAL_EVAL_EPS), eval_seed_base=EVAL_SEED_BASE + 8000,
            roll_pitch_limit_deg=FINAL_EVAL_RPL, coupling_scale=FINAL_EVAL_COUP,
            stability_weight=FINAL_EVAL_SW, ci=95.0)
        print(f"  Symbolic RMS@H: {sym_result['summaries'].get('rms_yaw_horizon', {}).get('mean', np.nan):.4f}")
    except Exception as e:
        print(f"  [SymbolicDistill] Failed: {e!r}")

    # 5. DAgger with KAN
    try:
        from morphing_glider.interpretability.kan import KANPolicyNetwork
        from morphing_glider.interpretability.dagger import DAggerDistillation
        print(f"\n{'=' * 60}\n5. DAgger + KAN\n{'=' * 60}")
        kan_device = torch.device(DEVICE)
        kan_student = KANPolicyNetwork(
            obs_dim=OBS_DIM, action_dim=6,
            hidden_dim=24 if FAST_DEV_RUN else 32,
            n_bases=5 if FAST_DEV_RUN else 6).to(kan_device)
        dagger = DAggerDistillation(
            expert=interp_ctrl, student=kan_student,
            n_iterations=3 if FAST_DEV_RUN else (6 if MEDIUM_RUN else 8),
            episodes_per_iter=2 if FAST_DEV_RUN else 3,
            max_steps=MAX_STEPS_EP, mix_probability=0.9, beta_decay=0.8, learning_rate=5e-4)
        dagger_history = dagger.train(verbose=True)
        DAggerDistillation.plot_training_history(
            dagger_history, save_path="morphing_glider_figures/dagger_training.png")
        kan_result = summarize_controller_over_episodes_bca(
            kan_student, label="KAN-DAgger", domain_scale=0.5, max_steps=MAX_STEPS_EP,
            eval_episodes=min(10, FINAL_EVAL_EPS), eval_seed_base=EVAL_SEED_BASE + 9000,
            roll_pitch_limit_deg=FINAL_EVAL_RPL, coupling_scale=FINAL_EVAL_COUP,
            stability_weight=FINAL_EVAL_SW, ci=95.0)
        print(f"  KAN-DAgger RMS@H: {kan_result['summaries'].get('rms_yaw_horizon', {}).get('mean', np.nan):.4f}")
        n_params = sum(p.numel() for p in kan_student.parameters())
        print(f"  Parameters: {n_params} (compression: {78000 / max(n_params, 1):.1f}x)")
    except Exception as e:
        print(f"  [DAgger] Failed: {e!r}")

    try:
        del interp_model
    except Exception:
        pass
    gc.collect()


if __name__ == "__main__":
    main()
