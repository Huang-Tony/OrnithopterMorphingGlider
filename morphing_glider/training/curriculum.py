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
                          eval_episodes, eval_seed_base, use_residual,
                          total_budget: Optional[int] = None, stop_on_gate_fail: bool = True,
                          callback=None, resume_dir: Optional[str] = None, pause_file: Optional[str] = None):
    """Gated multi-phase SAC curriculum.

    Revision R1 additions:
        total_budget: if given, the whole curriculum is limited to this many environment steps. Each
            phase may use at most an equal share of the budget still remaining (unused budget rolls
            over to later phases; the final phase receives everything left), so curriculum and flat
            SAC can be compared at an identical step budget.
        stop_on_gate_fail: original behaviour (True) ends the curriculum when a gate is missed without
            sufficient improvement. With False the curriculum advances to the next phase once its budget
            share is exhausted (budget-based progression), so the full budget is always used.
        resume_dir / pause_file: pause-and-resume support (see morphing_glider.training.resume).
    """
    # Deferred imports: evaluation subpackage and SB3 utilities
    from morphing_glider.evaluation import evaluate_controller, summarize_metrics
    from morphing_glider.config import DEVICE
    from morphing_glider.training.resume import (PauseRequested, PauseCheckCallback, pause_flag_set,
                                                 save_snapshot, load_snapshot)
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList

    assert phases, "Need at least one phase"
    logs: Dict[str, Any] = {"evaltrace": [], "eval_details": [], "phase_boundaries": [],
            "phases": [p.__dict__ for p in phases], "use_residual": bool(use_residual),
            "algo_name": "residual_curriculum" if use_residual else "curriculum", "resumes": []}
    vec = vecnorm = model = None

    # Compute max residual limit across all phases so the action_space stays fixed
    max_residual_limit = None
    if use_residual:
        all_lims = [p.residual_limit for p in phases if p.residual_limit is not None]
        if all_lims:
            max_residual_limit = np.max(np.stack([np.asarray(l, dtype=float) for l in all_lims]), axis=0)

    pause_cb = PauseCheckCallback(pause_file) if pause_file else None
    cbs = [c for c in (callback, pause_cb) if c is not None]
    learn_cb = CallbackList(cbs) if len(cbs) > 1 else (cbs[0] if cbs else None)
    snap = load_snapshot(resume_dir)
    L: Dict[str, Any] = {}   # loop state that is snapshotted

    def snapshot(reason):
        if not resume_dir: return
        L["obs_rms"] = copy.deepcopy(vecnorm.obs_rms) if vecnorm is not None else None
        L["lr"] = float(model.actor.optimizer.param_groups[0]["lr"])
        L["logs"] = logs
        if callback is not None and hasattr(callback, "records"): L["diag_records"] = callback.records
        save_snapshot(resume_dir, model=model, state=dict(L))
        print(f"  [SNAPSHOT:{reason}] phase={L.get('i')} phase_steps={L.get('phase_steps')} t={model.num_timesteps}")

    try:
        if snap is None:
            vec, vecnorm = build_training_env_for_phase(phases[0], seed=seed+2000, n_envs=n_envs,
                                                         max_steps=max_steps, prev_obs_rms=None,
                                                         use_residual=use_residual,
                                                         max_residual_limit=max_residual_limit)
            apply_phase_runtime_settings(vec, phases[0])
            model = build_sac_model(vec, seed=seed,
                                    tensorboard_log="tb_curriculum_residual" if use_residual else "tb_curriculum",
                                    learning_rate=float(phases[0].learning_rate))
            i0 = 0
        else:
            L.update(snap["state"]); logs = L["logs"]; i0 = int(L["i"])
            ph = phases[min(i0, len(phases) - 1)]          # i0 == len(phases): all phases already done
            logs.setdefault("resumes", []).append({"global_steps": None, "phase": ph.name})
            vec, vecnorm = build_training_env_for_phase(ph, seed=seed+2000+1000*i0+7919*len(logs["resumes"]),
                                                         n_envs=n_envs, max_steps=max_steps,
                                                         prev_obs_rms=L.get("obs_rms"), use_residual=use_residual,
                                                         max_residual_limit=max_residual_limit)
            if vecnorm is not None and L.get("obs_rms") is not None:
                vecnorm.obs_rms = copy.deepcopy(L["obs_rms"])     # undo the warm-up drift
            apply_phase_runtime_settings(vec, ph)
            model = SAC.load(snap["model"], env=vec, device=DEVICE)
            model.load_replay_buffer(snap["replay"])
            if callback is not None and hasattr(callback, "records"):
                callback.records = list(L.get("diag_records", []))
            logs["resumes"][-1]["global_steps"] = int(model.num_timesteps)
            print(f"[RESUME] curriculum from phase {i0} ({ph.name}), phase_steps={L.get('phase_steps')}, "
                  f"t={model.num_timesteps}")

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
                                          global_steps=int(model.num_timesteps), stats=stats,
                                          eval_rand_scale=eval_scale,
                                          eval_rpl=float(phase.roll_pitch_limit_deg))
            return stats

        if snap is None:
            print("\n" + "#"*80)
            print(f"CURRICULUM START | residual={use_residual} | seed={seed} | envs={n_envs}")
            print("Phases:", [p.name for p in phases])
            print("#"*80)
            init_stats = eval_hard("init", phases[0], residual_limit_eval=phases[0].residual_limit)
            L["init_steady"] = float(init_stats.get("mean_rms_yaw_steady", np.nan))
            L["prev_obs_rms"] = vecnorm.obs_rms if vecnorm else None
            L["phase_ready"] = False

        for i in range(i0, len(phases)):
            phase = phases[i]
            resuming_phase = (snap is not None and i == i0 and bool(L.get("phase_ready")))
            if not resuming_phase:
                print(f"\n{'='*80}\nPHASE {i+1}/{len(phases)} :: {phase.name}")
                print(f" twist={phase.twist_factor} rand={phase.rand_scale} steps={phase.max_timesteps} lr={phase.learning_rate:.2e}")
                print(f"{'='*80}")
                if i > 0 and not (snap is not None and i == i0):
                    try: vec.close()
                    except Exception: pass
                    del vec; gc.collect()
                    vec, vecnorm = build_training_env_for_phase(phase, seed=seed+2000+1000*i, n_envs=n_envs,
                        max_steps=max_steps, prev_obs_rms=L.get("prev_obs_rms"), use_residual=use_residual,
                        max_residual_limit=max_residual_limit)
                    apply_phase_runtime_settings(vec, phase)
                    model.set_env(vec)
                if i > 0:
                    _partial_replay_reset(model, retain_fraction=REPLAY_RETAIN_FRACTION)
                    init_stats_p = eval_hard("phase_init", phase, residual_limit_eval=phase.residual_limit)
                    L["init_steady"] = float(init_stats_p.get("mean_rms_yaw_steady", np.nan))
                L["prev_obs_rms"] = vecnorm.obs_rms if vecnorm else L.get("prev_obs_rms")
            if use_residual and phase.residual_limit is not None:
                _apply_residual_limit_on_vec(vec, phase.residual_limit)
            if callback is not None and hasattr(callback, 'set_phase'): callback.set_phase(phase.name)

            if not resuming_phase:
                _set_phase_lr_on_sac(model, float(phase.learning_rate), phase.name)
                max_ts = int(phase.max_timesteps)
                if total_budget is not None:
                    used = int(sum(int(b.get("phase_steps", 0)) for b in logs["phase_boundaries"]))
                    remaining = max(0, int(total_budget) - used)
                    phases_left = len(phases) - i
                    share = remaining if phases_left <= 1 else remaining // phases_left
                    max_ts = int(min(max_ts, max(share, 0)))
                    print(f"  [BUDGET] used={used} remaining={remaining} -> phase cap={max_ts}")
                L.update(i=i, phase_ready=True, phase_steps=0, max_ts=max_ts, chunk_end=None, gate_passed=False,
                         best_steady=float("inf"), best_steady_at=0, last_stats=None,
                         best_hirmssteady=float("inf"), best_policy_state=None, lr_decayed=False)
            else:
                _set_phase_lr_on_sac(model, float(L["lr"]), phase.name + "_resume")
            if int(L["max_ts"]) <= 0:
                logs["phase_boundaries"].append({"phase": phase.name, "global_steps": int(model.num_timesteps),
                                                 "phase_steps": 0, "gate_passed": False})
                L.update(i=i+1, phase_ready=False); continue

            chunk = int(eval_every_steps); max_ts = int(L["max_ts"])
            while L["phase_steps"] < max_ts:
                if L.get("chunk_end") is None:
                    L["chunk_end"] = int(min(L["phase_steps"] + chunk, max_ts))
                train_steps = int(max(1, L["chunk_end"] - L["phase_steps"]))
                t0 = time.time(); before = int(model.num_timesteps)
                try: model.learn(total_timesteps=train_steps, reset_num_timesteps=False, progress_bar=False, callback=learn_cb)
                except TypeError: model.learn(total_timesteps=train_steps, reset_num_timesteps=False, callback=learn_cb)
                wall = time.time() - t0; L["phase_steps"] += int(model.num_timesteps) - before
                if pause_flag_set(pause_file) and (pause_cb is None or pause_cb.paused or L["phase_steps"] < L["chunk_end"]):
                    snapshot("pause"); raise PauseRequested(f"paused in phase {phase.name}")
                L["chunk_end"] = None
                L["prev_obs_rms"] = vecnorm.obs_rms if vecnorm else L.get("prev_obs_rms")
                last_stats = eval_hard(f"{L['phase_steps']}", phase, residual_limit_eval=phase.residual_limit)
                L["last_stats"] = last_stats
                current_steady = float(last_stats.get("mean_rms_yaw_steady", np.nan))
                current_hirmssteady = float(last_stats.get("hirmssteady", np.nan))
                if np.isfinite(current_steady) and current_steady < L["best_steady"]:
                    L["best_steady"] = current_steady; L["best_steady_at"] = int(L["phase_steps"])
                if np.isfinite(current_hirmssteady) and current_hirmssteady < L["best_hirmssteady"]:
                    L["best_hirmssteady"] = current_hirmssteady
                    L["best_policy_state"] = (
                        copy.deepcopy(model.actor.state_dict()),
                        copy.deepcopy(model.critic.state_dict()),
                        copy.deepcopy(model.critic_target.state_dict()),
                        copy.deepcopy(vecnorm.obs_rms) if vecnorm else None,
                    )
                    print(f"  [BEST] New best hirmssteady={L['best_hirmssteady']:.4f} at step {L['phase_steps']}")
                if np.isfinite(current_steady) and np.isfinite(L["best_steady"]) and L["best_steady"] < float("inf"):
                    if current_steady > L["best_steady"] * 1.30:
                        print(f"  [REGRESSION WARNING] steady RMS {current_steady:.4f} > 1.30× best {L['best_steady']:.4f}")
                        if not L["lr_decayed"]:
                            new_lr = float(phase.learning_rate) * 0.5
                            _set_phase_lr_on_sac(model, new_lr, f"{phase.name}_decay")
                            L["lr_decayed"] = True
                print(f" [TRAIN] phase_steps={L['phase_steps']}/{max_ts} | wall={wall:.1f}s")

                gate_break = False
                if phase.target_rms is not None and L["phase_steps"] >= int(max(0, phase.min_steps_before_gate)):
                    target = float(phase.target_rms)
                    hirmssteady = float(last_stats.get("hirmssteady", np.nan))
                    gate_ok = np.isfinite(hirmssteady) and hirmssteady < target
                    gate_fr = float(last_stats.get("failure_rate", 1.0))
                    _max_fail = GATE_MAX_FAILURE_RATE_BY_PHASE.get(phase.name, GATE_MAX_FAILURE_RATE)
                    if gate_fr > _max_fail:
                        gate_ok = False
                        print(f"  [STABILITY BLOCK] failure={gate_fr*100:.1f}% > {_max_fail*100:.0f}%")
                    if gate_ok:
                        L["gate_passed"] = True; gate_break = True
                        print(f" [GATE PASS] hirmssteady={hirmssteady:.4f} < target={target:.4f}")
                    else:
                        print(f" [GATE FAIL] hirmssteady={hirmssteady:.4f} >= target={target:.4f}")
                snapshot("chunk")
                if gate_break: break

            # ---- Best-checkpoint restoration ----
            last_stats = L.get("last_stats")
            if not L["gate_passed"] and L["best_policy_state"] is not None:
                actor_sd, critic_sd, ct_sd, obs_rms_best = L["best_policy_state"]
                model.actor.load_state_dict(actor_sd)
                model.critic.load_state_dict(critic_sd)
                model.critic_target.load_state_dict(ct_sd)
                if vecnorm is not None and obs_rms_best is not None:
                    vecnorm.obs_rms = obs_rms_best
                print(f"  [RESTORE] Restored best checkpoint (hirmssteady={L['best_hirmssteady']:.4f})")
                last_stats = eval_hard("best_restored", phase, residual_limit_eval=phase.residual_limit)
                if phase.target_rms is not None:
                    h_restored = float(last_stats.get("hirmssteady", np.nan))
                    fr_restored = float(last_stats.get("failure_rate", 1.0))
                    _max_fail = GATE_MAX_FAILURE_RATE_BY_PHASE.get(phase.name, GATE_MAX_FAILURE_RATE)
                    if np.isfinite(h_restored) and h_restored < float(phase.target_rms) and fr_restored <= _max_fail:
                        L["gate_passed"] = True
                        print(f" [GATE PASS via BEST RESTORE] hirmssteady={h_restored:.4f} < target={phase.target_rms:.4f}")

            gate_passed = bool(L["gate_passed"]); best_steady = float(L["best_steady"]); init_steady = float(L["init_steady"])
            logs["phase_boundaries"].append({"phase": phase.name, "global_steps": int(model.num_timesteps),
                                             "phase_steps": int(L["phase_steps"]), "gate_passed": gate_passed})
            threshold = float(GATE_OVERRIDE_IMPROVEMENT_THRESHOLD_RESIDUAL if use_residual else GATE_OVERRIDE_IMPROVEMENT_THRESHOLD)
            improvement = float((init_steady - best_steady) / max(init_steady, 1e-6)) if np.isfinite(best_steady) and np.isfinite(init_steady) else float("nan")

            print(f"\n[TRAINING HEALTH] Phase={phase.name}")
            print(f"  Init steady={init_steady:.4f} Best={best_steady:.4f} Improvement={improvement*100:.1f}%")
            print(f"  Gate: {'PASSED' if gate_passed else 'NOT MET'}")

            stop = False
            if phase.target_rms is not None and not gate_passed:
                end_fr = float(last_stats.get("failure_rate", 1.0)) if last_stats else 1.0
                improved_enough = bool(np.isfinite(improvement) and improvement >= threshold)
                _max_fail = GATE_MAX_FAILURE_RATE_BY_PHASE.get(phase.name, GATE_MAX_FAILURE_RATE)
                if end_fr > _max_fail:
                    print(f"  [STABILITY BLOCK] end failure={end_fr*100:.1f}% > {_max_fail*100:.0f}%")
                    improved_enough = False
                if improved_enough:
                    print(f"  [GATE OVERRIDE] {improvement*100:.1f}% improvement; advancing.")
                elif not stop_on_gate_fail:
                    print(f"  [BUDGET ADVANCE] phase budget exhausted without gate pass; advancing.")
                else:
                    print(f"  [CURRICULUM STOP]"); stop = True
            L.update(i=i+1, phase_ready=False, best_policy_state=None)
            snapshot("phase_end")
            if stop: break

        print("\n" + "#"*80 + "\nCURRICULUM COMPLETE\n" + "#"*80)
        return model, vecnorm, logs
    finally:
        try:
            if vec is not None: vec.close()
        except Exception: pass
        gc.collect()
