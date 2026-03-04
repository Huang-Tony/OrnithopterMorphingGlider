"""Controller evaluation, hierarchical bootstrap summary, and eval-trace helpers."""

import gc
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from morphing_glider.config import DEVICE, DEFAULT_YAW_TARGETS
from morphing_glider.utils.numeric import bootstrap_mean_ci_bca, hierarchical_bootstrap_mean_ci
from morphing_glider.evaluation.metrics import run_episode, compute_episode_metrics, summarize_metrics


# ================================================================
# Core controller evaluation
# ================================================================

def evaluate_controller(controller, *, n_episodes, eval_seed_base, domain_rand_scale,
                        max_steps, twist_factor, use_residual_env, residual_limit=None,
                        store_histories=True, roll_pitch_limit_deg=70.0, coupling_scale=1.0,
                        stability_weight=0.03, sensor_noise_scale=1.0):
    from morphing_glider.training.infrastructure import make_env
    from morphing_glider.environment.wrappers import ProgressiveTwistWrapper, ResidualHeuristicWrapper
    from morphing_glider.controllers.heuristic import VirtualTendonHeuristicController

    mets = []; hists = []
    phase = {"name": "eval", "twist_factor": float(twist_factor)}
    for i in range(int(n_episodes)):
        seed = int(eval_seed_base + i)
        env = make_env(seed=seed, domain_rand_scale=float(domain_rand_scale),
                       max_steps=int(max_steps), for_eval=True, twist_enabled=True, include_omega_cross=True,
                       roll_pitch_limit_deg=float(roll_pitch_limit_deg), coupling_scale=float(coupling_scale),
                       stability_weight=float(stability_weight), sensor_noise_scale=float(sensor_noise_scale))
        if use_residual_env:
            heur = VirtualTendonHeuristicController(yaw_rate_max=max(abs(v) for v in DEFAULT_YAW_TARGETS))
            lim = residual_limit if residual_limit is not None else 0.08
            env = ResidualHeuristicWrapper(env, heuristic=heur, residual_limit=lim)
        env = ProgressiveTwistWrapper(env, phase=phase, twist_factor=float(twist_factor), reward_shaper=None, ramp_steps=0)
        hist = run_episode(env, controller, deterministic=True, seed=seed, max_steps=max_steps)
        met = compute_episode_metrics(hist, horizon_T=int(max_steps))
        mets.append(met)
        if store_histories: hists.append(hist)
    return mets, hists


# ================================================================
# BCa summary helpers
# ================================================================

def _bca_summary(values, *, ci=95.0, seed=0):
    v = np.asarray(values, dtype=float); v = v[np.isfinite(v)]
    if v.size == 0: return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "std": float("nan"), "n": 0.0}
    mean, lo, hi = bootstrap_mean_ci_bca(v, ci=ci, seed=seed)
    return {"mean": float(mean), "lo": float(lo), "hi": float(hi),
            "std": float(np.std(v, ddof=1)), "n": float(v.size)}


def _mean_of_metric(metrics, key):
    a = np.array([m.get(key, np.nan) for m in metrics], dtype=float); a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


# ================================================================
# Model-run evaluation (loads SAC checkpoint)
# ================================================================

def eval_model_run_metrics(rr, *, domain_scale, max_steps, eval_episodes, eval_seed_base,
                           roll_pitch_limit_deg, coupling_scale, stability_weight,
                           residual_limit=None, sensor_noise_scale=1.0):
    from stable_baselines3 import SAC
    from morphing_glider.training.infrastructure import load_vecnorm_for_eval
    from morphing_glider.controllers.sb3_controller import SB3Controller

    model = SAC.load(rr.model_path, device=DEVICE)
    vecnorm = load_vecnorm_for_eval(rr.vecnorm_path, max_steps=int(max_steps))
    ctrl = SB3Controller(model, obs_rms=(vecnorm.obs_rms if vecnorm else None),
                         clip_obs=(vecnorm.clip_obs if vecnorm else 10.0))
    mets, _ = evaluate_controller(ctrl, n_episodes=int(eval_episodes), eval_seed_base=int(eval_seed_base),
        domain_rand_scale=float(domain_scale), max_steps=int(max_steps), twist_factor=1.0,
        use_residual_env=(rr.algo_name == "residual_curriculum"), residual_limit=residual_limit,
        store_histories=False, roll_pitch_limit_deg=float(roll_pitch_limit_deg),
        coupling_scale=float(coupling_scale), stability_weight=float(stability_weight),
        sensor_noise_scale=float(sensor_noise_scale))
    try: del model
    except Exception: pass
    gc.collect()
    return mets


# ================================================================
# Eval metric keys
# ================================================================

EVAL_METRIC_KEYS = [
    "rms_yaw_horizon", "mae_yaw", "rms_yaw_steady", "rms_yaw_transient",
    "failure", "mean_settle_time", "mean_action_norm", "mean_power_loss",
    "mean_speed", "min_speed", "mean_vz", "mean_altitude", "delta_altitude",
]


# ================================================================
# Hierarchical bootstrap across training seeds
# ================================================================

def summarize_trained_algo_hierarchical(runs, *, algo_name, domain_scale, max_steps,
                                         eval_episodes, eval_seed_base, roll_pitch_limit_deg,
                                         coupling_scale, stability_weight, residual_limit=None, ci=95.0):
    """Evaluate trained algo across seeds using hierarchical bootstrap.

    Args:
        runs: List of TrainRunResult.
        algo_name: Algorithm name to filter.
        domain_scale: Eval domain randomization.
        max_steps: Episode max steps.
        eval_episodes: Episodes per seed.
        eval_seed_base: Base seed for eval.
        roll_pitch_limit_deg: Termination limit.
        coupling_scale: Coupling scale.
        stability_weight: Stability weight.
        residual_limit: Residual action limit.
        ci: Confidence level.

    Returns:
        Dict with algo_name, summaries (hierarchical CIs), seed_episode_data.

    References:
        [DAVISON_HINKLEY_1997] Bootstrap Methods.
    """
    seed_episodes: Dict[int, List[Dict[str, float]]] = {}
    for rr in runs:
        if rr.algo_name != algo_name: continue
        mets = eval_model_run_metrics(rr, domain_scale=float(domain_scale), max_steps=int(max_steps),
            eval_episodes=int(eval_episodes), eval_seed_base=int(eval_seed_base),
            roll_pitch_limit_deg=float(roll_pitch_limit_deg), coupling_scale=float(coupling_scale),
            stability_weight=float(stability_weight), residual_limit=residual_limit)
        seed_episodes[int(rr.train_seed)] = mets

    if not seed_episodes:
        return {"algo_name": algo_name, "domain_scale": float(domain_scale),
                "n_train_seeds": 0, "summaries": {}, "seed_episodes": {}}

    summaries = {}
    for k in EVAL_METRIC_KEYS:
        s2v = {}
        for s, mets in seed_episodes.items():
            s2v[s] = [float(m.get(k, np.nan)) for m in mets]
        mean, lo, hi = hierarchical_bootstrap_mean_ci(s2v, ci=ci, seed=100+(hash(k)%10000))
        all_vals = []
        for v in s2v.values(): all_vals.extend(v)
        std = float(np.nanstd(all_vals))
        summaries[k] = {"mean": float(mean), "lo": float(lo), "hi": float(hi), "std": std, "n": float(len(all_vals))}

    return {"algo_name": algo_name, "domain_scale": float(domain_scale),
            "n_train_seeds": int(len(seed_episodes)), "summaries": summaries,
            "seed_episodes": seed_episodes}


# ================================================================
# BCA summary over episodes for a single controller
# ================================================================

def summarize_controller_over_episodes_bca(controller, *, label, domain_scale, max_steps,
                                            eval_episodes, eval_seed_base, roll_pitch_limit_deg,
                                            coupling_scale, stability_weight, ci=95.0,
                                            use_residual_env=False, residual_limit=None,
                                            return_raw_metrics=False, sensor_noise_scale=1.0):
    mets, _ = evaluate_controller(controller, n_episodes=int(eval_episodes),
        eval_seed_base=int(eval_seed_base), domain_rand_scale=float(domain_scale),
        max_steps=int(max_steps), twist_factor=1.0, use_residual_env=bool(use_residual_env),
        residual_limit=residual_limit, store_histories=False,
        roll_pitch_limit_deg=float(roll_pitch_limit_deg), coupling_scale=float(coupling_scale),
        stability_weight=float(stability_weight), sensor_noise_scale=float(sensor_noise_scale))
    summaries = {}
    for k in EVAL_METRIC_KEYS:
        vals = [float(m.get(k, np.nan)) for m in mets]
        summaries[k] = _bca_summary(vals, ci=ci, seed=200+(hash(k)%10000))
    out = {"label": str(label), "domain_scale": float(domain_scale),
           "n_episodes": int(eval_episodes), "summaries": summaries}
    if return_raw_metrics: out["raw_metrics"] = mets
    return out


# ================================================================
# Final eval table
# ================================================================

def print_final_eval_table(eval_blocks, *, ci=95.0):
    """Print final eval table with Effect Size (d) column."""
    METS = [("rms_yaw_horizon","RMS@H"), ("rms_yaw_steady","RMS_ss"), ("failure","FailRate"),
            ("mean_settle_time","Settle"), ("mean_action_norm","ActNorm"), ("mean_power_loss","PwrLoss")]
    print("\n" + "="*120)
    print(f"FINAL EVAL TABLE (BCa {ci:.0f}% CI) + Effect Size (Cohen d vs Heuristic)")
    print("="*120)
    header = ["Algo", "Cond"] + [m[1] for m in METS] + ["d(RMS@H)"]
    print(" | ".join([f"{h:<18s}" if j<2 else f"{h:>12s}" for j,h in enumerate(header)]))
    print("-"*120)

    # Find heuristic RMS@H for effect size calculation
    heur_rmsh = {}; heur_std = {}
    for algo, cond, block in eval_blocks:
        if algo.lower() == "heuristic":
            s = block.get("summaries", {}).get("rms_yaw_horizon", {})
            heur_rmsh[cond] = float(s.get("mean", np.nan))
            heur_std[cond] = float(s.get("std", np.nan))

    def fmt(s, is_rate=False):
        mean = float(s.get("mean", np.nan)); lo = float(s.get("lo", np.nan)); hi = float(s.get("hi", np.nan))
        if not np.isfinite(mean): return "       n/a"
        if is_rate: return f"{mean*100:5.1f}[{lo*100:4.1f},{hi*100:4.1f}]"
        return f"{mean:6.3f}[{lo:5.3f},{hi:5.3f}]"

    for algo, cond, block in eval_blocks:
        sums = block.get("summaries", {})
        row = [f"{algo:<18s}", f"{cond:<18s}"]
        for key, _ in METS:
            s = sums.get(key, {})
            row.append(f"{fmt(s, is_rate=(key=='failure')):>12s}")
        # Cohen's d vs heuristic (pooled std)
        my_rms = float(sums.get("rms_yaw_horizon", {}).get("mean", np.nan))
        h_rms = heur_rmsh.get(cond, float("nan"))
        my_std = float(sums.get("rms_yaw_horizon", {}).get("std", np.nan))
        h_sd = heur_std.get(cond, float("nan"))
        if np.isfinite(my_rms) and np.isfinite(h_rms) and np.isfinite(my_std) and np.isfinite(h_sd):
            s_pooled = float(np.sqrt((my_std**2 + h_sd**2) / 2.0))
            if s_pooled > 1e-9:
                d = (my_rms - h_rms) / s_pooled
                row.append(f"{d:>+8.3f}")
            else:
                row.append(f"{'n/a':>12s}")
        else:
            row.append(f"{'n/a':>12s}")
        print(" | ".join(row))


# ================================================================
# Eval-trace append helper
# ================================================================

def _standardize_evaltrace_append(logs, *, tag, phase_name, global_steps, stats):
    entry = {"tag": str(tag), "phase": str(phase_name), "global_steps": int(global_steps),
             "mean_rmsh": float(stats.get("mean_rmsh", np.nan)),
             "lo_rmsh": float(stats.get("lo_rmsh", np.nan)), "hi_rmsh": float(stats.get("hi_rmsh", np.nan))}
    logs.setdefault("evaltrace", []).append(entry)
    logs.setdefault("eval_details", []).append({**entry, **stats})
