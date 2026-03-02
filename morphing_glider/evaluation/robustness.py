"""Robustness and out-of-distribution evaluation functions."""

from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from morphing_glider.utils.numeric import rms
from morphing_glider.evaluation.metrics import run_episode, compute_episode_metrics
from morphing_glider.evaluation.evaluate import (
    _bca_summary, evaluate_controller, EVAL_METRIC_KEYS,
)


def eval_ood_yaw_targets(policy, env_factory: Callable, targets: Tuple[float,...] = (-1.0, -0.8, 0.8, 1.0),
                         seeds: Sequence[int] = (0,), n_episodes: int = 10) -> Dict[str, Any]:
    from morphing_glider.training.infrastructure import make_env
    from morphing_glider.environment.wrappers import ProgressiveTwistWrapper

    results = {}
    for tgt in targets:
        all_rms = []; all_surv = []
        for sd in seeds:
            for ep in range(n_episodes):
                env = make_env(seed=int(sd*1000+ep), domain_rand_scale=0.5, max_steps=200,
                               for_eval=True, roll_pitch_limit_deg=65.0, coupling_scale=1.0, stability_weight=0.03)
                env.unwrapped.yaw_targets = [float(tgt)]
                env = ProgressiveTwistWrapper(env, phase={"name":"ood"}, twist_factor=1.0, reward_shaper=None)
                hist = run_episode(env, policy, deterministic=True, seed=int(sd*1000+ep))
                met = compute_episode_metrics(hist, horizon_T=200)
                all_rms.append(float(met.get("rms_yaw_horizon", np.nan)))
                all_surv.append(1.0 - float(met.get("failure", 1.0)))
        rms_ci = _bca_summary(all_rms); surv_ci = _bca_summary(all_surv)
        results[f"target_{tgt:.1f}"] = {"rms": rms_ci, "survival": surv_ci}
    return {"ood_targets": list(targets), "results": results}


def eval_distribution_shift(policy, env_factory: Callable, dr_scale: float = 1.5,
                            seeds: Sequence[int] = (0,), n_episodes: int = 10) -> Dict[str, Any]:
    from morphing_glider.training.infrastructure import make_env
    from morphing_glider.environment.wrappers import ProgressiveTwistWrapper

    all_rms = []; all_fail = []
    for sd in seeds:
        for ep in range(n_episodes):
            env = make_env(seed=int(sd*1000+ep), domain_rand_scale=float(dr_scale), max_steps=200,
                           for_eval=True, roll_pitch_limit_deg=65.0, coupling_scale=1.0, stability_weight=0.03)
            env = ProgressiveTwistWrapper(env, phase={"name":"dr_shift"}, twist_factor=1.0, reward_shaper=None)
            hist = run_episode(env, policy, deterministic=True, seed=int(sd*1000+ep))
            met = compute_episode_metrics(hist, horizon_T=200)
            all_rms.append(float(met.get("rms_yaw_horizon", np.nan)))
            all_fail.append(float(met.get("failure", 1.0)))
    return {"dr_scale": dr_scale, "rms": _bca_summary(all_rms), "failure": _bca_summary(all_fail)}


def eval_sensor_corruption(policy, env_factory: Callable, noise_mult: float = 3.0,
                           seeds: Sequence[int] = (0,), n_episodes: int = 10) -> Dict[str, Any]:
    from morphing_glider.training.infrastructure import make_env
    from morphing_glider.environment.wrappers import ProgressiveTwistWrapper

    all_rms = []; all_fail = []
    for sd in seeds:
        for ep in range(n_episodes):
            env = make_env(seed=int(sd*1000+ep), domain_rand_scale=1.0, max_steps=200,
                           for_eval=True, roll_pitch_limit_deg=65.0, coupling_scale=1.0,
                           stability_weight=0.03, sensor_noise_scale=float(noise_mult))
            env = ProgressiveTwistWrapper(env, phase={"name":"noisy"}, twist_factor=1.0, reward_shaper=None)
            hist = run_episode(env, policy, deterministic=True, seed=int(sd*1000+ep))
            met = compute_episode_metrics(hist, horizon_T=200)
            all_rms.append(float(met.get("rms_yaw_horizon", np.nan)))
            all_fail.append(float(met.get("failure", 1.0)))
    return {"noise_mult": noise_mult, "rms": _bca_summary(all_rms), "failure": _bca_summary(all_fail)}


def eval_long_horizon(policy, env_factory: Callable, max_steps: int = 750,
                      seeds: Sequence[int] = (0,), n_episodes: int = 10) -> Dict[str, Any]:
    from morphing_glider.training.infrastructure import make_env
    from morphing_glider.environment.wrappers import ProgressiveTwistWrapper

    all_surv = []; all_alt_loss = []; all_rms_late = []; all_max_rp = []
    for sd in seeds:
        for ep in range(n_episodes):
            env = make_env(seed=int(sd*1000+ep), domain_rand_scale=1.0, max_steps=int(max_steps),
                           for_eval=True, roll_pitch_limit_deg=65.0, coupling_scale=1.0, stability_weight=0.03)
            env = ProgressiveTwistWrapper(env, phase={"name":"long"}, twist_factor=1.0, reward_shaper=None)
            hist = run_episode(env, policy, deterministic=True, seed=int(sd*1000+ep), max_steps=int(max_steps))
            survived = len(hist) >= int(max_steps)
            all_surv.append(float(survived))
            if len(hist) > 1:
                alt = np.array([h["altitude"] for h in hist])
                all_alt_loss.append(float(alt[0] - alt[-1]))
                roll_arr = np.array([abs(h["roll"]) for h in hist])
                pitch_arr = np.array([abs(h["pitch"]) for h in hist])
                all_max_rp.append(float(max(np.max(roll_arr), np.max(pitch_arr))))
                if len(hist) > 200:
                    late_err = np.array([h["yaw_rate"]-h["yaw_ref"] for h in hist[200:]])
                    all_rms_late.append(rms(late_err))
    return {"max_steps": max_steps, "survival_750": _bca_summary(all_surv),
            "altitude_loss_m": _bca_summary(all_alt_loss),
            "rms_200_750": _bca_summary(all_rms_late),
            "max_roll_pitch_rad": _bca_summary(all_max_rp)}


def eval_mid_episode_parameter_jump(policy, env_factory: Callable,
                                     seeds: Sequence[int] = (0,), n_episodes: int = 10) -> Dict[str, Any]:
    from morphing_glider.training.infrastructure import make_env
    from morphing_glider.environment.wrappers import ProgressiveTwistWrapper
    from morphing_glider.physics.domain_randomizer import DomainRandomizer

    recovery_times = []
    for sd in seeds:
        for ep in range(n_episodes):
            env = make_env(seed=int(sd*1000+ep), domain_rand_scale=0.5, max_steps=300,
                           for_eval=True, roll_pitch_limit_deg=65.0, coupling_scale=1.0, stability_weight=0.03)
            env = ProgressiveTwistWrapper(env, phase={"name":"jump"}, twist_factor=1.0, reward_shaper=None)
            obs, info = env.reset(seed=int(sd*1000+ep))
            if hasattr(policy, "reset"): policy.reset()
            base = env.unwrapped.unwrapped if hasattr(env.unwrapped, "unwrapped") else env.unwrapped
            rng_jump = np.random.default_rng(int(sd*1000+ep+7777))
            recovered = False; rec_step = float("nan")
            for t in range(300):
                if t == 100:
                    new_phys = DomainRandomizer(enabled=True, scale=1.0).sample(rng_jump)
                    base.phys = new_phys
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if t > 100 and not recovered:
                    if abs(float(info.get("yaw_error", 1.0))) < 0.05:
                        recovered = True; rec_step = float(t - 100)
                if terminated or truncated: break
            recovery_times.append(rec_step if recovered else float("nan"))
    return {"recovery_steps": _bca_summary(recovery_times),
            "recovered_fraction": float(np.mean([1.0 if np.isfinite(r) else 0.0 for r in recovery_times]))}


def model_quality_ceiling(policy, *, max_steps: int = 200, n_episodes: int = 20,
                          eval_seed_base: int = 88888) -> Dict[str, Any]:
    mets, _ = evaluate_controller(policy, n_episodes=n_episodes, eval_seed_base=eval_seed_base,
        domain_rand_scale=0.0, max_steps=max_steps, twist_factor=1.0, use_residual_env=False,
        store_histories=False, roll_pitch_limit_deg=65.0, coupling_scale=1.0,
        stability_weight=0.03, sensor_noise_scale=0.0)
    summaries = {}
    for k in EVAL_METRIC_KEYS:
        vals = [float(m.get(k, np.nan)) for m in mets]
        summaries[k] = _bca_summary(vals)
    return {"label": "oracle_zero_noise_zero_DR", "summaries": summaries}
