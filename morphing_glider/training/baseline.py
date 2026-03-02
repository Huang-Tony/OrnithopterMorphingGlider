"""Flat (non-curriculum) SAC training."""

import gc, time
from typing import Any, Dict

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
                       eval_every_steps, eval_episodes, eval_seed_base, eval_domain_rand_scale=1.0):
    # Deferred import: evaluation subpackage
    from morphing_glider.evaluation import evaluate_controller, summarize_metrics

    phase = PhaseSpec(name="baseline_full_twist", twist_factor=1.0, rand_scale=1.0,
                      max_timesteps=int(total_timesteps), ramp_steps=0, reward_shaper=None,
                      roll_pitch_limit_deg=70.0, coupling_scale=1.0, stability_weight=0.03)
    logs: Dict[str, Any] = {"evaltrace": [], "eval_details": [], "phase_boundaries": [],
            "phases": [phase.__dict__], "use_residual": False, "algo_name": "baseline"}
    vec = vecnorm = model = None
    try:
        vec, vecnorm = build_training_env_for_phase(phase, seed=seed+1000, n_envs=n_envs,
                                                     max_steps=max_steps, prev_obs_rms=None, use_residual=False)
        apply_phase_runtime_settings(vec, phase)
        model = build_sac_model_baseline(vec, seed=seed, tensorboard_log="tb_baseline")

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
                                          global_steps=int(model.num_timesteps), stats=stats)
            return stats

        print(f"\n--- TRAIN BASELINE SAC | seed={seed} | envs={n_envs} | steps={total_timesteps} ---")
        eval_now("init")
        trained = 0; first = True
        while trained < int(total_timesteps):
            chunk = int(min(int(eval_every_steps), int(total_timesteps) - trained))
            t0 = time.time()
            try: model.learn(total_timesteps=chunk, reset_num_timesteps=bool(first), progress_bar=False)
            except TypeError: model.learn(total_timesteps=chunk, reset_num_timesteps=bool(first))
            first = False; trained += chunk; wall = time.time() - t0
            eval_now(str(trained))
            print(f" [TRAIN] baseline_steps={trained}/{int(total_timesteps)} | wall={wall:.1f}s")
        return model, vecnorm, logs
    finally:
        try:
            if vec is not None: vec.close()
        except Exception: pass
        gc.collect()
