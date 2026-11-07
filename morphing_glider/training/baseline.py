"""Flat (non-curriculum) SAC training."""

import gc, time
from typing import Any, Dict, Optional

import numpy as np

from morphing_glider.config import FAST_DEV_RUN
from morphing_glider.controllers.sb3_controller import SB3Controller
from morphing_glider.training.infrastructure import (
    build_training_env_for_phase,
    apply_phase_runtime_settings,
    build_sac_model_baseline,
    _standardize_evaltrace_append,
)
from morphing_glider.training.curriculum import PhaseSpec

# Deferred imports resolved at call time to avoid circular dependencies
# with evaluation module: evaluate_controller, summarize_metrics


def train_baseline_sac(*, total_timesteps, seed, n_envs, max_steps,
                       eval_every_steps, eval_episodes, eval_seed_base, eval_domain_rand_scale=1.0,
                       checkpoint_dir=None, callback=None,
                       resume_dir: Optional[str] = None, pause_file: Optional[str] = None):
    """Flat SAC. Revision R1 additions: checkpoint_dir saves the policy (and VecNormalize stats) before
    training and after every evaluation chunk (training-progress videos); callback is passed to
    model.learn (training diagnostics); resume_dir/pause_file enable pause-and-resume."""
    # Deferred imports: evaluation subpackage and resume utilities
    import copy, os
    from morphing_glider.evaluation import evaluate_controller, summarize_metrics
    from morphing_glider.config import DEVICE
    from morphing_glider.training.resume import (PauseRequested, PauseCheckCallback, pause_flag_set,
                                                 save_snapshot, load_snapshot)
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList

    phase = PhaseSpec(name="baseline_full_twist", twist_factor=1.0, rand_scale=1.0,
                      max_timesteps=int(total_timesteps), ramp_steps=0, reward_shaper=None,
                      roll_pitch_limit_deg=70.0, coupling_scale=1.0, stability_weight=0.03)
    logs: Dict[str, Any] = {"evaltrace": [], "eval_details": [], "phase_boundaries": [],
            "phases": [phase.__dict__], "use_residual": False, "algo_name": "baseline", "resumes": []}
    pause_cb = PauseCheckCallback(pause_file) if pause_file else None
    cbs = [c for c in (callback, pause_cb) if c is not None]
    learn_cb = CallbackList(cbs) if len(cbs) > 1 else (cbs[0] if cbs else None)
    snap = load_snapshot(resume_dir)
    S: Dict[str, Any] = {"trained": 0, "first": True, "chunk_end": None}
    vec = vecnorm = model = None
    try:
        if snap is None:
            vec, vecnorm = build_training_env_for_phase(phase, seed=seed+1000, n_envs=n_envs,
                                                         max_steps=max_steps, prev_obs_rms=None, use_residual=False)
            apply_phase_runtime_settings(vec, phase)
            model = build_sac_model_baseline(vec, seed=seed, tensorboard_log="tb_baseline")
        else:
            S.update(snap["state"]); logs = S["logs"]
            vec, vecnorm = build_training_env_for_phase(phase, seed=seed+1000+7919*(len(logs.get("resumes", []))+1),
                                                         n_envs=n_envs, max_steps=max_steps,
                                                         prev_obs_rms=S.get("obs_rms"), use_residual=False)
            if vecnorm is not None and S.get("obs_rms") is not None:
                vecnorm.obs_rms = copy.deepcopy(S["obs_rms"])
            apply_phase_runtime_settings(vec, phase)
            model = SAC.load(snap["model"], env=vec, device=DEVICE)
            model.load_replay_buffer(snap["replay"])
            if callback is not None and hasattr(callback, "records"):
                callback.records = list(S.get("diag_records", []))
            logs.setdefault("resumes", []).append({"global_steps": int(model.num_timesteps)})
            print(f"[RESUME] baseline at trained={S['trained']} (t={model.num_timesteps})")

        def snapshot(reason):
            if not resume_dir: return
            S["obs_rms"] = copy.deepcopy(vecnorm.obs_rms) if vecnorm is not None else None
            S["logs"] = logs
            if callback is not None and hasattr(callback, "records"): S["diag_records"] = callback.records
            save_snapshot(resume_dir, model=model, state=dict(S))
            print(f"  [SNAPSHOT:{reason}] trained={S['trained']} t={model.num_timesteps}")

        def eval_now(tag):
            ctrl = SB3Controller(model, obs_rms=(vecnorm.obs_rms if vecnorm else None),
                                 clip_obs=(vecnorm.clip_obs if vecnorm else 10.0))
            mets, _ = evaluate_controller(ctrl, n_episodes=int(eval_episodes),
                eval_seed_base=int(eval_seed_base), domain_rand_scale=float(eval_domain_rand_scale),
                max_steps=int(max_steps), twist_factor=1.0, use_residual_env=False, store_histories=False,
                roll_pitch_limit_deg=float(phase.roll_pitch_limit_deg),
                coupling_scale=float(phase.coupling_scale), stability_weight=float(phase.stability_weight))
            stats = summarize_metrics(mets, label=f"EVAL({tag}) baseline | rand={eval_domain_rand_scale}",
                                      ci_method="bca", print_cost_terms=True)
            _standardize_evaltrace_append(logs, tag=str(tag), phase_name=phase.name,
                                          global_steps=int(model.num_timesteps), stats=stats,
                                          eval_rand_scale=float(eval_domain_rand_scale),
                                          eval_rpl=float(phase.roll_pitch_limit_deg))
            return stats

        def checkpoint(step):
            if not checkpoint_dir: return
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save(os.path.join(checkpoint_dir, f"step_{int(step):07d}.zip"))
            if vecnorm is not None: vecnorm.save(os.path.join(checkpoint_dir, f"step_{int(step):07d}.vecnorm.pkl"))

        if snap is None:
            print(f"\n--- TRAIN BASELINE SAC | seed={seed} | envs={n_envs} | steps={total_timesteps} ---")
            eval_now("init"); checkpoint(0)
        while S["trained"] < int(total_timesteps):
            if S.get("chunk_end") is None:
                S["chunk_end"] = int(min(S["trained"] + int(eval_every_steps), int(total_timesteps)))
            chunk = int(max(1, S["chunk_end"] - S["trained"]))
            t0 = time.time(); before = int(model.num_timesteps)
            try: model.learn(total_timesteps=chunk, reset_num_timesteps=bool(S["first"]), progress_bar=False, callback=learn_cb)
            except TypeError: model.learn(total_timesteps=chunk, reset_num_timesteps=bool(S["first"]), callback=learn_cb)
            S["first"] = False; S["trained"] += int(model.num_timesteps) - before; wall = time.time() - t0
            logs.setdefault("wall_clock_s", []).append(float(wall))
            if pause_flag_set(pause_file) and (pause_cb is None or pause_cb.paused or S["trained"] < S["chunk_end"]):
                snapshot("pause"); raise PauseRequested("paused baseline")
            S["chunk_end"] = None
            eval_now(str(S["trained"])); checkpoint(S["trained"])
            snapshot("chunk")
            print(f" [TRAIN] baseline_steps={S['trained']}/{int(total_timesteps)} | wall={wall:.1f}s")
        return model, vecnorm, logs
    finally:
        try:
            if vec is not None: vec.close()
        except Exception: pass
        gc.collect()
