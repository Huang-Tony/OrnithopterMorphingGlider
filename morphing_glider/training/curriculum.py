"""Curriculum training: PhaseSpec, TrainRunResult, and multi-phase SAC training."""

import copy, gc, time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from morphing_glider.config import (
    CURRICULUM_EVAL_RAND_PAD, REPLAY_RETAIN_FRACTION, DEFAULT_YAW_TARGETS,
    GATE_MAX_FAILURE_RATE, GATE_MAX_FAILURE_RATE_BY_PHASE,
    GATE_OVERRIDE_IMPROVEMENT_THRESHOLD, GATE_OVERRIDE_IMPROVEMENT_THRESHOLD_RESIDUAL,
)
from morphing_glider.environment.wrappers import mild_curriculum_reward_shaper
from morphing_glider.controllers.sb3_controller import SB3Controller
from morphing_glider.training.infrastructure import (
    build_training_env_for_phase,
    apply_phase_runtime_settings,
    build_sac_model,
    _set_phase_lr_on_sac,
    _standardize_evaltrace_append,
    _partial_replay_reset,
    _apply_residual_limit_on_vec,
)


# ================================================================
# Data classes
# ================================================================

@dataclass
class PhaseSpec:
    name: str; twist_factor: float; rand_scale: float; max_timesteps: int
    ramp_steps: int = 0; start_twist_factor: Optional[float] = None
    reward_shaper: Optional[Callable] = mild_curriculum_reward_shaper
    residual_limit: Optional[Union[float, np.ndarray]] = None
    learning_rate: float = 3e-4; target_rms: Optional[float] = None
    min_steps_before_gate: int = 0; roll_pitch_limit_deg: float = 70.0
    coupling_scale: float = 1.0; stability_weight: float = 0.03


@dataclass
class TrainRunResult:
    algo_name: str; train_seed: int; model_path: str
    vecnorm_path: Optional[str]; train_logs: Dict[str, Any]


# ================================================================
# Curriculum progression summary
# ================================================================

def summarize_curriculum_progression(phase_logs: Dict[str, Any]) -> None:
    """Print a table of phase | n_steps | gate_passed | failure_rate | rms_steady.

    Args:
        phase_logs: Training logs dict with evaltrace and phase_boundaries.

    Returns:
        None. Prints formatted table.

    References:
        [BENGIO_2009] Curriculum Learning.
    """
    print("\n" + "="*80)
    print("CURRICULUM PROGRESSION SUMMARY")
    print("="*80)
    print(f"  {'Phase':<20s} {'Steps':>8s} {'Gate':>8s} {'FailRate':>10s} {'RMS_ss':>10s}")
    print("-"*80)
    boundaries = phase_logs.get("phase_boundaries", [])
    details = phase_logs.get("eval_details", [])
    for b in boundaries:
        pname = b.get("phase", "?"); gsteps = b.get("global_steps", 0)
        last_detail = None
        for d in reversed(details):
            if d.get("phase", "") == pname:
                last_detail = d; break
        if last_detail:
            fr = float(last_detail.get("failure_rate", np.nan))
            rms_ss = float(last_detail.get("mean_rms_yaw_steady", np.nan))
            gate = "Y" if last_detail.get("hirmssteady", 999) < 999 else "?"
        else:
            fr = float("nan"); rms_ss = float("nan"); gate = "?"
        print(f"  {pname:<20s} {gsteps:>8d} {gate:>8s} {fr*100:>9.1f}% {rms_ss:>10.4f}")


# ================================================================
# Curriculum training
# ================================================================

def train_with_curriculum(*, phases, seed, n_envs, max_steps, eval_every_steps,
                          eval_episodes, eval_seed_base, use_residual):
    # Deferred import: evaluation subpackage
    from morphing_glider.evaluation import evaluate_controller, summarize_metrics

    assert phases, "Need at least one phase"
    logs: Dict[str, Any] = {"evaltrace": [], "eval_details": [], "phase_boundaries": [],
            "phases": [p.__dict__ for p in phases], "use_residual": bool(use_residual),
            "algo_name": "residual_curriculum" if use_residual else "curriculum"}
    vec = vecnorm = model = None

    # Compute max residual limit across all phases so the action_space stays fixed
    max_residual_limit = None
    if use_residual:
        all_lims = [p.residual_limit for p in phases if p.residual_limit is not None]
        if all_lims:
            max_residual_limit = np.max(np.stack([np.asarray(l, dtype=float) for l in all_lims]), axis=0)

    try:
        vec, vecnorm = build_training_env_for_phase(phases[0], seed=seed+2000, n_envs=n_envs,
                                                     max_steps=max_steps, prev_obs_rms=None,
                                                     use_residual=use_residual,
                                                     max_residual_limit=max_residual_limit)
        apply_phase_runtime_settings(vec, phases[0])
        model = build_sac_model(vec, seed=seed,
                                tensorboard_log="tb_curriculum_residual" if use_residual else "tb_curriculum",
                                learning_rate=float(phases[0].learning_rate))

        def eval_hard(tag, phase, residual_limit_eval=None):
            eval_scale = float(min(1.0, float(phase.rand_scale) + CURRICULUM_EVAL_RAND_PAD))
            ctrl = SB3Controller(model, obs_rms=(vecnorm.obs_rms if vecnorm else None),
                                 clip_obs=(vecnorm.clip_obs if vecnorm else 10.0))
            mets, _ = evaluate_controller(ctrl, n_episodes=int(eval_episodes), eval_seed_base=int(eval_seed_base),
                domain_rand_scale=float(eval_scale), max_steps=int(max_steps), twist_factor=1.0,
                use_residual_env=bool(use_residual), residual_limit=residual_limit_eval, store_histories=False,
                roll_pitch_limit_deg=float(phase.roll_pitch_limit_deg),
                coupling_scale=float(phase.coupling_scale), stability_weight=float(phase.stability_weight))
            stats = summarize_metrics(mets,
                label=f"EVAL({tag}) {phase.name} | rand={eval_scale:.2f} | rp={phase.roll_pitch_limit_deg:.0f}°",
                ci_method="bca", print_cost_terms=True)
            _standardize_evaltrace_append(logs, tag=str(tag), phase_name=str(phase.name),
                                          global_steps=int(model.num_timesteps), stats=stats)
            return stats

        print("\n" + "#"*80)
        print(f"CURRICULUM START | residual={use_residual} | seed={seed} | envs={n_envs}")
        print("Phases:", [p.name for p in phases])
        print("#"*80)

        init_stats = eval_hard("init", phases[0], residual_limit_eval=phases[0].residual_limit)
        init_steady = float(init_stats.get("mean_rms_yaw_steady", np.nan))
        prev_obs_rms = vecnorm.obs_rms if vecnorm else None

        for i, phase in enumerate(phases):
            print(f"\n{'='*80}\nPHASE {i+1}/{len(phases)} :: {phase.name}")
            print(f" twist={phase.twist_factor} rand={phase.rand_scale} steps={phase.max_timesteps} lr={phase.learning_rate:.2e}")
            print(f"{'='*80}")

            if i > 0:
                try: vec.close()
                except Exception: pass
                del vec; gc.collect()
                vec, vecnorm = build_training_env_for_phase(phase, seed=seed+2000+1000*i, n_envs=n_envs,
                    max_steps=max_steps, prev_obs_rms=prev_obs_rms, use_residual=use_residual,
                    max_residual_limit=max_residual_limit)
                apply_phase_runtime_settings(vec, phase)
                model.set_env(vec)
                _partial_replay_reset(model, retain_fraction=REPLAY_RETAIN_FRACTION)
                init_stats_p = eval_hard("phase_init", phase, residual_limit_eval=phase.residual_limit)
                init_steady = float(init_stats_p.get("mean_rms_yaw_steady", np.nan))

            prev_obs_rms = vecnorm.obs_rms if vecnorm else prev_obs_rms
            if use_residual and phase.residual_limit is not None:
                _apply_residual_limit_on_vec(vec, phase.residual_limit)
            _set_phase_lr_on_sac(model, float(phase.learning_rate), phase.name)

            phase_steps = 0; max_ts = int(phase.max_timesteps); chunk = int(eval_every_steps)
            gate_passed = False; best_steady = float("inf"); best_steady_at = 0; last_stats = None
            best_hirmssteady = float("inf"); best_policy_state = None; lr_decayed = False

            while phase_steps < max_ts:
                train_steps = int(min(chunk, max_ts - phase_steps))
                t0 = time.time()
                try: model.learn(total_timesteps=train_steps, reset_num_timesteps=False, progress_bar=False)
                except TypeError: model.learn(total_timesteps=train_steps, reset_num_timesteps=False)
                wall = time.time() - t0; phase_steps += train_steps
                prev_obs_rms = vecnorm.obs_rms if vecnorm else prev_obs_rms
                last_stats = eval_hard(f"{phase_steps}", phase, residual_limit_eval=phase.residual_limit)
                current_steady = float(last_stats.get("mean_rms_yaw_steady", np.nan))
                current_hirmssteady = float(last_stats.get("hirmssteady", np.nan))
                if np.isfinite(current_steady) and current_steady < best_steady:
                    best_steady = current_steady; best_steady_at = int(phase_steps)
                if np.isfinite(current_hirmssteady) and current_hirmssteady < best_hirmssteady:
                    best_hirmssteady = current_hirmssteady
                    best_policy_state = (
                        copy.deepcopy(model.actor.state_dict()),
                        copy.deepcopy(model.critic.state_dict()),
                        copy.deepcopy(model.critic_target.state_dict()),
                        copy.deepcopy(vecnorm.obs_rms) if vecnorm else None,
                    )
                    print(f"  [BEST] New best hirmssteady={best_hirmssteady:.4f} at step {phase_steps}")
                if np.isfinite(current_steady) and np.isfinite(best_steady) and best_steady < float("inf"):
                    if current_steady > best_steady * 1.30:
                        print(f"  [REGRESSION WARNING] steady RMS {current_steady:.4f} > 1.30× best {best_steady:.4f}")
                        if not lr_decayed:
                            new_lr = float(phase.learning_rate) * 0.5
                            _set_phase_lr_on_sac(model, new_lr, f"{phase.name}_decay")
                            lr_decayed = True
                print(f" [TRAIN] phase_steps={phase_steps}/{max_ts} | wall={wall:.1f}s")

                if phase.target_rms is not None and phase_steps >= int(max(0, phase.min_steps_before_gate)):
                    target = float(phase.target_rms)
                    hirmssteady = float(last_stats.get("hirmssteady", np.nan))
                    gate_ok = np.isfinite(hirmssteady) and hirmssteady < target
                    gate_fr = float(last_stats.get("failure_rate", 1.0))
                    _max_fail = GATE_MAX_FAILURE_RATE_BY_PHASE.get(phase.name, GATE_MAX_FAILURE_RATE)
                    if gate_fr > _max_fail:
                        gate_ok = False
                        print(f"  [STABILITY BLOCK] failure={gate_fr*100:.1f}% > {_max_fail*100:.0f}%")
                    if gate_ok:
                        gate_passed = True
                        print(f" [GATE PASS] hirmssteady={hirmssteady:.4f} < target={target:.4f}"); break
                    else:
                        print(f" [GATE FAIL] hirmssteady={hirmssteady:.4f} >= target={target:.4f}")

            # ---- Best-checkpoint restoration ----
            if not gate_passed and best_policy_state is not None:
                actor_sd, critic_sd, ct_sd, obs_rms_best = best_policy_state
                model.actor.load_state_dict(actor_sd)
                model.critic.load_state_dict(critic_sd)
                model.critic_target.load_state_dict(ct_sd)
                if vecnorm is not None and obs_rms_best is not None:
                    vecnorm.obs_rms = obs_rms_best
                print(f"  [RESTORE] Restored best checkpoint (hirmssteady={best_hirmssteady:.4f})")
                last_stats = eval_hard("best_restored", phase, residual_limit_eval=phase.residual_limit)
                if phase.target_rms is not None:
                    h_restored = float(last_stats.get("hirmssteady", np.nan))
                    fr_restored = float(last_stats.get("failure_rate", 1.0))
                    _max_fail = GATE_MAX_FAILURE_RATE_BY_PHASE.get(phase.name, GATE_MAX_FAILURE_RATE)
                    if np.isfinite(h_restored) and h_restored < float(phase.target_rms) and fr_restored <= _max_fail:
                        gate_passed = True
                        print(f" [GATE PASS via BEST RESTORE] hirmssteady={h_restored:.4f} < target={phase.target_rms:.4f}")

            logs["phase_boundaries"].append({"phase": phase.name, "global_steps": int(model.num_timesteps)})
            threshold = float(GATE_OVERRIDE_IMPROVEMENT_THRESHOLD_RESIDUAL if use_residual else GATE_OVERRIDE_IMPROVEMENT_THRESHOLD)
            improvement = float((init_steady - best_steady) / max(init_steady, 1e-6)) if np.isfinite(best_steady) and np.isfinite(init_steady) else float("nan")

            print(f"\n[TRAINING HEALTH] Phase={phase.name}")
            print(f"  Init steady={init_steady:.4f} Best={best_steady:.4f} Improvement={improvement*100:.1f}%")
            print(f"  Gate: {'PASSED' if gate_passed else 'NOT MET'}")

            if phase.target_rms is not None and not gate_passed:
                end_fr = float(last_stats.get("failure_rate", 1.0)) if last_stats else 1.0
                improved_enough = bool(np.isfinite(improvement) and improvement >= threshold)
                _max_fail = GATE_MAX_FAILURE_RATE_BY_PHASE.get(phase.name, GATE_MAX_FAILURE_RATE)
                if end_fr > _max_fail:
                    print(f"  [STABILITY BLOCK] end failure={end_fr*100:.1f}% > {_max_fail*100:.0f}%")
                    improved_enough = False
                if improved_enough:
                    print(f"  [GATE OVERRIDE] {improvement*100:.1f}% improvement; advancing.")
                else:
                    print(f"  [CURRICULUM STOP]"); break

        print("\n" + "#"*80 + "\nCURRICULUM COMPLETE\n" + "#"*80)
        return model, vecnorm, logs
    finally:
        try:
            if vec is not None: vec.close()
        except Exception: pass
        gc.collect()
