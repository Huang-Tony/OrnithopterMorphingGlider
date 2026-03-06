"""Environment creation, model building, checkpointing, and training utilities."""

import os, math, gc, json, time
from typing import Any, Dict, Callable, Optional, Union

import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from morphing_glider.config import (
    DEVICE, DT, GLOBAL_SEED, USE_VECNORMALIZE, USE_SDE,
    VECENV_MODE, SUBPROC_START_METHOD, FAST_DEV_RUN, MEDIUM_RUN, PAPER_RUN,
    TRAIN_AERO_PANELS, EVAL_AERO_PANELS, HYPERPARAMETER_REGISTRY,
    CURRICULUM_EVAL_RAND_PAD, REPLAY_RETAIN_FRACTION, DEFAULT_YAW_TARGETS,
    GATE_MAX_FAILURE_RATE, GATE_MAX_FAILURE_RATE_BY_PHASE,
    GATE_OVERRIDE_IMPROVEMENT_THRESHOLD, GATE_OVERRIDE_IMPROVEMENT_THRESHOLD_RESIDUAL,
    HOLD_RANGE_STEPS, BEZIER_ITERS_TRAIN, BEZIER_ITERS_EVAL,
    REPRO_TOLERANCE,
)
from morphing_glider.environment.env import MorphingGliderEnv6DOF
from morphing_glider.environment.wrappers import (
    ProgressiveTwistWrapper, ResidualHeuristicWrapper,
    mild_curriculum_reward_shaper,
)
from morphing_glider.controllers.heuristic import VirtualTendonHeuristicController
from morphing_glider.controllers.sb3_controller import SB3Controller
from morphing_glider.environment.observation import OBS_DIM


# ================================================================
# Environment construction
# ================================================================

def make_env(*, seed, domain_rand_scale, max_steps, for_eval=False, twist_enabled=True,
             include_omega_cross=True, roll_pitch_limit_deg=70.0, coupling_scale=1.0,
             stability_weight=0.03, start_altitude=200.0, sensor_noise_scale=1.0,
             reward_computer=None):
    """Create a single MorphingGliderEnv6DOF with specified configuration.

    Args:
        seed: RNG seed.
        domain_rand_scale: Domain randomization scale [0,1].
        max_steps: Episode length.
        for_eval: Use eval-quality aero panels.
        twist_enabled: Allow wing twist.
        include_omega_cross: Include rotational velocity cross terms in aero.
        roll_pitch_limit_deg: Termination limit [deg].
        coupling_scale: Roll/pitch coupling scale.
        stability_weight: Stability shaping weight.
        start_altitude: Initial altitude [m].
        sensor_noise_scale: Multiplier on sensor noise.
        reward_computer: Optional custom RewardComputer.

    Returns:
        Configured gym.Env instance.

    References:
        [BROCKMAN_2016] OpenAI Gym.
    """
    num_panels = EVAL_AERO_PANELS if for_eval else TRAIN_AERO_PANELS
    env = MorphingGliderEnv6DOF(
        max_steps=int(max_steps), twist_enabled=bool(twist_enabled),
        include_omega_cross=bool(include_omega_cross), yaw_targets=DEFAULT_YAW_TARGETS,
        hold_range_steps=HOLD_RANGE_STEPS, num_aero_panels=num_panels,
        domain_rand_scale=float(domain_rand_scale), domain_rand_enabled=True,
        actuator_tau=0.07, start_altitude=float(start_altitude),
        speed_min_terminate=6.0, roll_pitch_limit_deg=float(roll_pitch_limit_deg),
        coupling_scale=float(coupling_scale), stability_weight=float(stability_weight),
        sensor_noise_scale=float(sensor_noise_scale),
        reward_computer=reward_computer,
        seed=int(seed),
    )
    iters = BEZIER_ITERS_EVAL if for_eval else BEZIER_ITERS_TRAIN
    try:
        env.unwrapped.spar_R.iterations = iters
        env.unwrapped.spar_L.iterations = iters
    except Exception:
        pass
    return env


def make_vec_env(env_fns, *, mode, start_method="spawn"):
    mode = str(mode).lower().strip()
    if mode == "dummy":
        print("[VecEnv] Using DummyVecEnv"); return DummyVecEnv(env_fns)
    if mode == "subproc":
        print(f"[VecEnv] Trying SubprocVecEnv(start_method={start_method!r}) ...")
        try: return SubprocVecEnv(env_fns, start_method=str(start_method))
        except Exception as e:
            print(f"[VecEnv] SubprocVecEnv failed: {e!r} → DummyVecEnv"); return DummyVecEnv(env_fns)
    if FAST_DEV_RUN:
        print("[VecEnv] AUTO => DummyVecEnv (FAST_DEV_RUN)"); return DummyVecEnv(env_fns)
    try: return SubprocVecEnv(env_fns, start_method=str(start_method))
    except Exception:
        print("[VecEnv] SubprocVecEnv failed → DummyVecEnv"); return DummyVecEnv(env_fns)


def _find_wrapper(env, wrapper_type):
    e = env
    while True:
        if isinstance(e, wrapper_type): return e
        if hasattr(e, "env"): e = e.env; continue
        return None


def warmup_vecnormalize(vec_env, *, n_steps=2000, use_residual_hint=None):
    if not isinstance(vec_env, VecNormalize) or int(n_steps) <= 0: return
    use_heur = False
    if use_residual_hint is not None: use_heur = bool(use_residual_hint)
    else:
        try:
            base = vec_env.venv
            if isinstance(base, DummyVecEnv) and base.envs:
                if _find_wrapper(base.envs[0], ResidualHeuristicWrapper) is not None: use_heur = True
        except Exception: pass
    obs = vec_env.reset()
    if use_heur:
        a0 = np.zeros((vec_env.num_envs,) + vec_env.action_space.shape, dtype=np.float32)
        for _ in range(int(n_steps)):
            obs, rew, done, info = vec_env.step(a0)
            if np.any(done): vec_env.reset()
    else:
        for _ in range(int(n_steps)):
            a = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])
            obs, rew, done, info = vec_env.step(a)
            if np.any(done): vec_env.reset()


def apply_phase_runtime_settings(vec_env, phase):
    rp = float(getattr(phase, "roll_pitch_limit_deg", 70.0))
    cs = float(getattr(phase, "coupling_scale", 1.0))
    sw = float(getattr(phase, "stability_weight", 0.03))
    try:
        base = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
        if hasattr(base, "envs"):
            for e in base.envs:
                b = e.unwrapped
                b.roll_pitch_limit = math.radians(rp)
                b.coupling_scale = float(np.clip(cs, 0.0, 1.0))
                b.stability_weight = float(max(0.0, sw))
            return
    except Exception:
        pass
    try:
        vec_env.env_method("set_roll_pitch_limit_deg", rp)
        vec_env.env_method("set_coupling_scale", cs)
        vec_env.env_method("set_stability_weight", sw)
    except Exception as e:
        print(f"[PhaseConfig] WARNING: {e!r}")


# ================================================================
# SAC model construction
# ================================================================

def build_sac_model_baseline(vec_env, *, seed, tensorboard_log):
    action_dim = int(np.prod(vec_env.action_space.shape))
    policy_kwargs = dict(net_arch=[512, 256, 128], activation_fn=torch.nn.ReLU)
    return SAC("MlpPolicy", vec_env, seed=int(seed), device=DEVICE, verbose=0,
               buffer_size=150_000 if FAST_DEV_RUN else 500_000,
               learning_starts=2_000 if FAST_DEV_RUN else 8_000,
               batch_size=256 if not FAST_DEV_RUN else 128,
               learning_rate=3e-4, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=2,
               ent_coef="auto_0.1", target_entropy=-float(action_dim)*0.5, policy_kwargs=policy_kwargs,
               use_sde=bool(USE_SDE), sde_sample_freq=4 if USE_SDE else -1,
               tensorboard_log=str(tensorboard_log))


def build_sac_model(vec_env, *, seed, tensorboard_log, learning_rate=3e-4):
    action_dim = int(np.prod(vec_env.action_space.shape))
    policy_kwargs = dict(net_arch=[512, 256, 128], activation_fn=torch.nn.ReLU)
    return SAC("MlpPolicy", vec_env, seed=int(seed), device=DEVICE, verbose=0,
               buffer_size=150_000 if FAST_DEV_RUN else 500_000,
               learning_starts=2_000 if FAST_DEV_RUN else 8_000,
               batch_size=256 if not FAST_DEV_RUN else 128,
               learning_rate=float(learning_rate), tau=0.005, gamma=0.99,
               train_freq=1, gradient_steps=2, ent_coef="auto_0.1",
               target_entropy=-float(action_dim)*0.5, policy_kwargs=policy_kwargs,
               use_sde=bool(USE_SDE), sde_sample_freq=4 if USE_SDE else -1,
               tensorboard_log=str(tensorboard_log))


def _set_phase_lr_on_sac(model, lr, phase_name):
    lr = float(lr)
    try:
        for opt in [model.actor.optimizer, model.critic.optimizer]:
            for g in opt.param_groups: g["lr"] = lr
    except Exception: pass
    try: model.learning_rate = lr; model.lr_schedule = (lambda _pr, _lr=lr: _lr)
    except Exception: pass
    try:
        if hasattr(model, "ent_coef_optimizer") and model.ent_coef_optimizer is not None:
            for g in model.ent_coef_optimizer.param_groups: g["lr"] = lr
    except Exception: pass
    print(f"  [LR] Set lr={lr:.2e} for phase {phase_name!r}")


# ================================================================
# Eval-trace helpers
# ================================================================

def _standardize_evaltrace_append(logs, *, tag, phase_name, global_steps, stats,
                                   eval_rand_scale=None, eval_rpl=None):
    entry = {"tag": str(tag), "phase": str(phase_name), "global_steps": int(global_steps),
             "mean_rmsh": float(stats.get("mean_rmsh", np.nan)),
             "lo_rmsh": float(stats.get("lo_rmsh", np.nan)), "hi_rmsh": float(stats.get("hi_rmsh", np.nan))}
    if eval_rand_scale is not None:
        entry["eval_rand_scale"] = float(eval_rand_scale)
    if eval_rpl is not None:
        entry["eval_rpl"] = float(eval_rpl)
    logs.setdefault("evaltrace", []).append(entry)
    logs.setdefault("eval_details", []).append({**entry, **stats})


# ================================================================
# Replay buffer management
# ================================================================

def _partial_replay_reset(model, retain_fraction=REPLAY_RETAIN_FRACTION):
    # Fix G: partial replay buffer retention
    retain_fraction = float(np.clip(retain_fraction, 0.0, 1.0))
    try:
        buf = getattr(model, "replay_buffer", None)
        if buf is None: return
        n = int(buf.size())
        if n <= 0: buf.reset(); return
        keep = int(math.floor(n * retain_fraction))
        if keep <= 0: buf.reset(); print("  [REPLAY] Full reset."); return
        if bool(getattr(buf, "full", False)):
            pos = int(getattr(buf, "pos", 0)); bs = int(getattr(buf, "buffer_size", n))
            idx_all = np.concatenate([np.arange(pos, bs, dtype=int), np.arange(0, pos, dtype=int)])
            idx_keep = idx_all[-keep:]
        else:
            idx_keep = np.arange(n - keep, n, dtype=int)
        for f in ["observations", "next_observations", "actions", "rewards", "dones", "timeouts"]:
            if not hasattr(buf, f): continue
            arr = getattr(buf, f)
            if arr is None: continue
            try: tmp = arr[idx_keep].copy(); arr[:keep] = tmp
            except Exception: continue
        buf.pos = int(keep); buf.full = False
        print(f"  [REPLAY] Retained {keep}/{n} transitions ({retain_fraction*100:.0f}%)")
    except Exception as e:
        try: getattr(model, "replay_buffer").reset()
        except Exception: pass
        print(f"  [REPLAY] WARNING: partial retention failed ({e!r}); did full reset.")


# ================================================================
# Residual limit propagation
# ================================================================

def _apply_residual_limit_on_vec(vec, lim):
    new_lim = np.asarray(lim, dtype=float)
    try: vec.env_method("set_residual_limit", new_lim); return
    except Exception: pass
    base = vec.venv if isinstance(vec, VecNormalize) else vec
    if isinstance(base, DummyVecEnv):
        for e in base.envs:
            rw = _find_wrapper(e, ResidualHeuristicWrapper)
            if rw is not None: rw.set_residual_limit(new_lim)


# ================================================================
# Training environment construction
# ================================================================

def build_training_env_for_phase(phase, *, seed, n_envs, max_steps, prev_obs_rms,
                                 use_residual, max_residual_limit=None):
    phase_dict = {"name": phase.name, "twist_factor": float(np.clip(phase.twist_factor, 0.0, 1.0)),
                  "rand_scale": float(np.clip(phase.rand_scale, 0.0, 1.0)),
                  "ramp_steps": int(max(0, phase.ramp_steps)),
                  "start_twist_factor": float(phase.start_twist_factor) if phase.start_twist_factor is not None else float(np.clip(phase.twist_factor, 0.0, 1.0)),
                  "reward_shaper": phase.reward_shaper}
    _max_rl = max_residual_limit  # capture for closure
    def thunk(rank):
        def _init():
            env = make_env(seed=int(seed+rank), domain_rand_scale=float(phase.rand_scale),
                           max_steps=int(max_steps), for_eval=False, twist_enabled=True, include_omega_cross=True,
                           roll_pitch_limit_deg=float(phase.roll_pitch_limit_deg),
                           coupling_scale=float(phase.coupling_scale), stability_weight=float(phase.stability_weight))
            env = Monitor(env)
            if use_residual:
                heur = VirtualTendonHeuristicController(yaw_rate_max=max(abs(v) for v in DEFAULT_YAW_TARGETS))
                lim = phase.residual_limit if phase.residual_limit is not None else 0.08
                env = ResidualHeuristicWrapper(env, heuristic=heur, residual_limit=lim,
                                              action_space_limit=_max_rl)
            env = ProgressiveTwistWrapper(env, phase=phase_dict, twist_factor=float(phase.twist_factor),
                                         reward_shaper=phase.reward_shaper, ramp_steps=int(phase.ramp_steps),
                                         start_twist_factor=phase.start_twist_factor)
            return env
        return _init
    env_fns = [thunk(i) for i in range(int(n_envs))]
    vec = make_vec_env(env_fns, mode=VECENV_MODE, start_method=SUBPROC_START_METHOD)
    vecnorm = None
    if USE_VECNORMALIZE:
        vecnorm = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
        if prev_obs_rms is not None: vecnorm.obs_rms = prev_obs_rms
        vec = vecnorm
        warmup_vecnormalize(vec, n_steps=1200 if FAST_DEV_RUN else 3500, use_residual_hint=use_residual)
    return vec, vecnorm


# ================================================================
# Model / VecNormalize persistence
# ================================================================

def save_model_and_vecnorm(model, vecnorm, *, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{name}.zip"); model.save(model_path)
    vecnorm_path = None
    if vecnorm is not None:
        vecnorm_path = os.path.join(out_dir, f"{name}.vecnorm.pkl"); vecnorm.save(vecnorm_path)
    return model_path, vecnorm_path


def load_vecnorm_for_eval(vecnorm_path, *, max_steps):
    if vecnorm_path is None or not os.path.exists(vecnorm_path): return None
    try:
        dummy = make_env(seed=GLOBAL_SEED+444, domain_rand_scale=0.0, max_steps=int(max_steps), for_eval=True)
        return VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: dummy]))
    except Exception: return None


# ================================================================
# Training checkpoints
# ================================================================

def save_training_checkpoint(model: SAC, path: str, metadata: Dict[str, Any]) -> None:
    """Save SB3 model and sidecar JSON with metadata.

    Args:
        model: Trained SAC model.
        path: File path for the .zip checkpoint.
        metadata: Dict with seed, step, phase, eval metrics, etc.

    Returns:
        None. Writes model.zip and model.meta.json.

    References:
        [RAFFIN_2021] Stable-Baselines3.
    """
    model.save(path)
    meta_path = path.replace(".zip", ".meta.json")
    try:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=float)
        print(f"  [Checkpoint] Saved: {path} + {meta_path}")
    except Exception as e:
        print(f"  [Checkpoint] Meta save failed: {e!r}")


def verify_checkpoint_reproducibility(path: str, env_factory: Callable, n_episodes: int = 10,
                                      tolerance: float = REPRO_TOLERANCE) -> bool:
    """Reload checkpoint and verify eval RMS within tolerance of logged value.

    Reads algo type from .meta.json to use matching eval conditions
    (obs normalization, domain randomization, residual wrapper).

    Args:
        path: Path to .zip checkpoint.
        env_factory: Callable returning a gym.Env (unused, kept for API compat).
        n_episodes: Number of verification episodes.
        tolerance: Allowed RMS deviation from logged value.

    Returns:
        True if verified within tolerance.

    References:
        [PINEAU_2021] Improving Reproducibility in ML.
    """
    from morphing_glider.evaluation import evaluate_controller

    meta_path = path.replace(".zip", ".meta.json")
    meta = {}
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        pass
    logged_rms = float(meta.get("mean_rmsh", np.nan))
    algo = str(meta.get("algo", "baseline"))
    is_residual = ("residual" in algo.lower())

    try:
        model = SAC.load(path, device=DEVICE)
        # Load VecNormalize stats for proper obs normalization
        vecnorm_path = meta.get("vecnorm_path") or path.replace(".zip", ".vecnorm.pkl")
        vn = load_vecnorm_for_eval(str(vecnorm_path), max_steps=200)
        ctrl = SB3Controller(model,
                             obs_rms=(vn.obs_rms if vn else None),
                             clip_obs=(vn.clip_obs if vn else 10.0))
        eval_rand = float(meta.get("eval_rand_scale", 1.0))
        eval_rpl = float(meta.get("eval_rpl", 70.0))
        mets, _ = evaluate_controller(ctrl, n_episodes=n_episodes, eval_seed_base=GLOBAL_SEED+99999,
                                       domain_rand_scale=eval_rand, max_steps=200, twist_factor=1.0,
                                       use_residual_env=is_residual, store_histories=False,
                                       roll_pitch_limit_deg=eval_rpl)
        rms_vals = [float(m.get("rms_yaw_horizon", np.nan)) for m in mets]
        current_rms = float(np.nanmean(rms_vals))
        del model; gc.collect()
        if np.isfinite(logged_rms) and np.isfinite(current_rms):
            passed = abs(current_rms - logged_rms) <= tolerance
            print(f"  [REPRO] algo={algo} rand={eval_rand} rpl={eval_rpl}° logged={logged_rms:.4f} current={current_rms:.4f} tol={tolerance} → {'PASS' if passed else 'FAIL'}")
            return passed
        print(f"  [REPRO] Cannot verify (logged={logged_rms}, current={current_rms})")
        return False
    except Exception as e:
        print(f"  [REPRO] Verification failed: {e!r}"); return False
