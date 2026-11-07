"""Evaluation with full data capture (revision R1).

Protocol identical to morphing_glider.evaluation.evaluate_controller: make_env(for_eval=True) (10 aero
panels, 12 Bezier iterations), ProgressiveTwistWrapper(twist=1, no shaping), deterministic policy,
episode seeds eval_seed_base+i, 200-step horizon, one common roll/pitch limit (65 deg) for all controllers.

Saved per (controller, condition[, seed]):
  <out>/<tag>.npz   per-step arrays [episode, step, ...] (NaN-padded after termination)
  <out>/<tag>.json  per-episode metrics (paper metrics + extended flight metrics), sampled physics
                    parameters of each episode (domain-randomisation draws), seeds, termination reasons.
For SB3 policies, the Gaussian policy mean (pre-squash) and std of every step are stored as well.
"""

import math
import os
import time

import numpy as np

import common  # noqa: F401
from common import (MAX_STEPS_EP, EVAL_RPL_DEG, EVAL_COUPLING, EVAL_STAB_W, dump_json)
from morphing_glider.config import DT
from morphing_glider.evaluation.metrics import compute_episode_metrics
from morphing_glider.training.infrastructure import make_env
from morphing_glider.environment.wrappers import ProgressiveTwistWrapper, ResidualHeuristicWrapper
from morphing_glider.utils.quaternion import quat_to_euler_xyz

STEP_KEYS = ["yaw_rate", "yaw_ref", "roll", "pitch", "yaw", "omega_p", "omega_q", "speed", "altitude", "vz_world",
             "moment_x", "moment_y", "moment_z", "drag_R", "drag_L", "power_loss_total", "struct_energy_sum",
             "struct_energy_norm", "z_asym", "z_sym", "reward", "tracking_reward", "survival_bonus", "wall_cost",
             "total_penalty", "cost_track", "cost_att", "cost_rates", "cost_ctrl", "cost_jerk", "cost_power",
             "cost_struct", "cost_zsym", "terminal_penalty"]


class PolicyProbe:
    """Wraps an SB3Controller; records the Gaussian policy mean/std for every predicted action."""

    def __init__(self, ctrl):
        self.ctrl = ctrl; self.last_mu = None; self.last_std = None

    def reset(self):
        if hasattr(self.ctrl, "reset"): self.ctrl.reset()

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        import torch
        obs_n = self.ctrl._normalize_obs(np.asarray(obs, dtype=np.float32))
        m = self.ctrl.model
        with torch.no_grad():
            t = torch.as_tensor(obs_n[None], device=m.device)
            mu, log_std, _ = m.actor.get_action_dist_params(t)
        self.last_mu = mu.cpu().numpy()[0]; self.last_std = log_std.exp().cpu().numpy()[0]
        return self.ctrl.predict(obs, state, episode_start, deterministic)


def make_eval_env(seed, domain_rand_scale, *, use_residual=False, residual_limit=None, rpl=EVAL_RPL_DEG,
                  sensor_noise_scale=1.0, phys_override=None, actuator_tau=None, physics_substeps=None):
    from morphing_glider.environment.env import MorphingGliderEnv6DOF
    env = make_env(seed=seed, domain_rand_scale=float(domain_rand_scale), max_steps=MAX_STEPS_EP, for_eval=True,
                   twist_enabled=True, include_omega_cross=True, roll_pitch_limit_deg=float(rpl),
                   coupling_scale=EVAL_COUPLING, stability_weight=EVAL_STAB_W, sensor_noise_scale=sensor_noise_scale)
    base = env.unwrapped
    if actuator_tau is not None: base.act_tau = float(actuator_tau)
    if physics_substeps is not None: base.physics_substeps = int(physics_substeps)
    if phys_override:   # applied after every reset (model-mismatch stress tests)
        orig_reset = base.reset
        def reset_with_override(seed=None, options=None, _o=orig_reset):
            obs, info = _o(seed=seed, options=options)
            for k, v in phys_override.items():
                base.phys[k] = float(v(base.phys[k])) if callable(v) else float(v)
            return base._get_obs(), info
        base.reset = reset_with_override
    if use_residual:
        from morphing_glider.controllers.heuristic import VirtualTendonHeuristicController
        env = ResidualHeuristicWrapper(env, heuristic=VirtualTendonHeuristicController(yaw_rate_max=0.6),
                                       residual_limit=residual_limit if residual_limit is not None else 0.08)
    return ProgressiveTwistWrapper(env, phase={"name": "eval", "twist_factor": 1.0}, twist_factor=1.0,
                                   reward_shaper=None, ramp_steps=0)


def run_episode_full(env, controller, seed, max_steps=MAX_STEPS_EP):
    obs, info0 = env.reset(seed=seed)
    if hasattr(controller, "reset"): controller.reset()
    base = env.unwrapped
    rec = {k: [] for k in STEP_KEYS}
    rec.update(action=[], p3_R=[], p3_L=[], p3_cmd_R=[], p3_cmd_L=[], spar_R_p1=[], spar_R_p2=[], spar_L_p1=[],
               spar_L_p2=[], gust=[], omega=[], pos=[], vel=[], policy_mu=[], policy_std=[], t_ctrl_s=[])
    hist = []
    for t in range(int(max_steps)):
        t0 = time.perf_counter()
        action, _ = controller.predict(obs, deterministic=True)
        rec["t_ctrl_s"].append(time.perf_counter() - t0)
        if isinstance(controller, PolicyProbe):
            rec["policy_mu"].append(controller.last_mu.copy()); rec["policy_std"].append(controller.last_std.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        roll, pitch, yaw = quat_to_euler_xyz(base.q)
        info = dict(info); info["reward"] = float(reward)
        for k in STEP_KEYS:
            rec[k].append(float(info.get(k, np.nan)) if np.isscalar(info.get(k, np.nan)) else np.nan)
        rec["action"].append(np.asarray(action, float).copy())
        rec["p3_R"].append(base.p3_R.copy()); rec["p3_L"].append(base.p3_L.copy())
        rec["p3_cmd_R"].append(base.p3_cmd_R.copy()); rec["p3_cmd_L"].append(base.p3_cmd_L.copy())
        rec["spar_R_p1"].append(base.spar_R.p1.copy()); rec["spar_R_p2"].append(base.spar_R.p2.copy())
        rec["spar_L_p1"].append(base.spar_L.p1.copy()); rec["spar_L_p2"].append(base.spar_L.p2.copy())
        rec["gust"].append(base.gust.copy()); rec["omega"].append(base.omega.copy())
        rec["pos"].append(base.pos_world.copy()); rec["vel"].append(base.vel_world.copy())
        hist.append({"t": t, "yaw_rate": float(info.get("yaw_rate")), "yaw_ref": float(info.get("yaw_ref")),
                     "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw), "speed": float(info.get("speed")),
                     "altitude": float(info.get("altitude")), "vz_world": float(info.get("vz_world")),
                     "reward": float(reward), "action": np.asarray(action, float).copy(),
                     "terminated": bool(terminated), "truncated": bool(truncated), "info": info})
        if terminated or truncated: break
    met = compute_episode_metrics(hist, horizon_T=int(max_steps))
    met.update(extended_metrics(rec, hist))
    met["termination_reason"] = str(hist[-1]["info"].get("termination_reason", "")) if hist else ""
    met["seed"] = int(seed)
    arrays = {k: np.asarray(v, dtype=float) for k, v in rec.items() if len(v)}
    return met, arrays, dict(info0.get("phys", {}))


def extended_metrics(rec, hist):
    """Flight-performance metrics requested in review (all over the survived part of the episode)."""
    dt = float(DT); out = {}
    T = len(hist); out["survival_time_s"] = T * dt
    roll = np.degrees(np.asarray(rec["roll"])); pitch = np.degrees(np.asarray(rec["pitch"]))
    p = np.degrees(np.asarray(rec["omega_p"])); q = np.degrees(np.asarray(rec["omega_q"]))
    out.update(rms_roll_deg=float(np.sqrt(np.mean(roll ** 2))), rms_pitch_deg=float(np.sqrt(np.mean(pitch ** 2))),
               max_abs_roll_deg=float(np.max(np.abs(roll))), max_abs_pitch_deg=float(np.max(np.abs(pitch))),
               rms_roll_rate_dps=float(np.sqrt(np.mean(p ** 2))), rms_pitch_rate_dps=float(np.sqrt(np.mean(q ** 2))))
    a = np.asarray(rec["action"]); tips = np.hstack([np.asarray(rec["p3_R"]), np.asarray(rec["p3_L"])])
    neutral = np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0])
    da = np.diff(np.vstack([np.zeros((1, 6)), a]), axis=0)
    dtip = np.diff(np.vstack([neutral[None], tips]), axis=0)
    out["mean_tip_speed_mps"] = float(np.mean(np.linalg.norm(dtip.reshape(-1, 2, 3), axis=2)) / dt)
    out["tip_travel_m_per_s"] = float(np.sum(np.linalg.norm(dtip.reshape(-1, 2, 3), axis=2)) / max(T * dt, dt))
    out["mean_sq_action_increment"] = float(np.mean(np.sum(da ** 2, axis=1)))
    out["rms_action_rate_mps"] = float(np.sqrt(np.mean(np.sum(da ** 2, axis=1))) / dt)
    if T >= 16:   # fraction of action-signal power above 2 Hz (Nyquist 12.5 Hz)
        A = np.abs(np.fft.rfft(a - a.mean(axis=0), axis=0)) ** 2; f = np.fft.rfftfreq(T, dt)
        tot = A[1:].sum(); out["hf_action_power_frac_2Hz"] = float(A[f > 2.0].sum() / tot) if tot > 0 else 0.0
    disp = np.linalg.norm((tips - neutral).reshape(-1, 2, 3), axis=2)
    out.update(mean_tip_displacement_m=float(disp.mean()), max_tip_displacement_m=float(disp.max()),
               mean_abs_sweep_m=float(np.mean(np.abs(tips[:, [0, 3]]))),
               mean_abs_dihedral_m=float(np.mean(np.abs(tips[:, [2, 5]]))),
               mean_abs_dz_asym_m=float(np.mean(np.abs(np.asarray(rec["z_asym"])))),
               mean_struct_energy_norm=float(np.nanmean(rec["struct_energy_norm"])))
    from morphing_glider.physics.domain_randomizer import NOMINAL_PHYS
    m, g = NOMINAL_PHYS["mass"], NOMINAL_PHYS["g"]
    h = np.asarray(rec["altitude"]); V = np.asarray(rec["speed"])
    E = m * g * h + 0.5 * m * V ** 2
    out.update(mean_aero_power_W=float(np.nanmean(rec["power_loss_total"])),
               mean_energy_loss_rate_W=float(-(E[-1] - E[0]) / max((T - 1) * dt, dt)) if T > 1 else float("nan"),
               mean_sink_rate_mps=float(-np.mean(rec["vz_world"])), mean_airspeed_mps=float(np.mean(V)),
               altitude_loss_m=float(h[0] - h[-1]))
    e = np.asarray(rec["yaw_rate"]) - np.asarray(rec["yaw_ref"])
    out["rms_yaw_survived"] = float(np.sqrt(np.mean(e ** 2)))
    out["mean_ctrl_latency_ms"] = float(1000 * np.mean(rec["t_ctrl_s"]))
    return out


def evaluate(controller, *, label, condition, rand_scale, n_episodes, seed_base, out_dir=None, tag=None,
             save_arrays=True, verbose=True, **env_kwargs):
    mets, phys, per_ep = [], [], []
    t0 = time.time()
    for i in range(int(n_episodes)):
        env = make_eval_env(seed_base + i, rand_scale, **env_kwargs)
        met, arr, ph = run_episode_full(env, controller, seed_base + i)
        mets.append(met); phys.append(ph); per_ep.append(arr)
        try: env.close()
        except Exception: pass
    res = {"label": label, "condition": condition, "rand_scale": rand_scale, "n_episodes": len(mets),
           "seed_base": seed_base, "episodes": mets, "physics_draws": phys, "env_kwargs": {k: str(v) for k, v in env_kwargs.items()},
           "wall_s": time.time() - t0}
    if out_dir and tag:
        dump_json(res, os.path.join(out_dir, f"{tag}.json"))
        if save_arrays:
            keys = sorted({k for a in per_ep for k in a})
            packed = {}
            for k in keys:
                shp = next(a[k].shape[1:] for a in per_ep if k in a)
                X = np.full((len(per_ep), MAX_STEPS_EP) + shp, np.nan, dtype=np.float32)
                for j, a in enumerate(per_ep):
                    if k in a: X[j, :a[k].shape[0]] = a[k]
                packed[k] = X
            np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"), **packed)
    if verbose:
        r = np.array([m["rms_yaw_horizon"] for m in mets]); f = np.array([m["failure"] for m in mets])
        print(f"  {label:<22} {condition:<10} RMS@H={r.mean():.3f}  fail={100*f.mean():5.1f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
    return res


def load_sac(model_path, device="cpu"):
    from stable_baselines3 import SAC
    from morphing_glider.training.infrastructure import load_vecnorm_for_eval
    from morphing_glider.controllers.sb3_controller import SB3Controller
    model = SAC.load(model_path, device=device)
    vn = load_vecnorm_for_eval(model_path.replace(".zip", ".vecnorm.pkl"), max_steps=MAX_STEPS_EP)
    return PolicyProbe(SB3Controller(model, obs_rms=vn.obs_rms if vn else None, clip_obs=vn.clip_obs if vn else 10.0))
