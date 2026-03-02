"""Environment wrappers: residual heuristic and progressive twist curriculum."""

import numpy as np
import gymnasium as gym

from morphing_glider.environment.env import YAW_REF_MAX


class ResidualHeuristicWrapper(gym.Wrapper):
    def __init__(self, env, *, heuristic, residual_limit=0.08):
        super().__init__(env); self.heuristic = heuristic
        self.action_space = self.env.action_space; self.observation_space = self.env.observation_space
        self.residual_limit = np.full((6,), 0.08, dtype=float); self.set_residual_limit(residual_limit)
        self._last_obs = None

    def set_residual_limit(self, lim):
        lim = np.asarray(lim, dtype=float)
        if lim.size == 1: lim = np.full((6,), float(lim.item()), dtype=float)
        self.residual_limit = lim.astype(float, copy=True)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = np.array(obs, copy=True); self.heuristic.reset()
        info = dict(info); info["residual_mode"] = True; info["residual_limit"] = self.residual_limit.copy()
        return obs, info

    def step(self, action):
        res = np.clip(np.asarray(action, dtype=float).reshape(-1), -self.residual_limit, self.residual_limit)
        h, _ = self.heuristic.predict(self._last_obs, deterministic=True)
        a = np.clip(np.asarray(h, dtype=float).reshape(-1) + res, self.env.action_space.low, self.env.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(a)
        self._last_obs = np.array(obs, copy=True)
        info = dict(info); info["heur_action_norm"] = float(np.linalg.norm(h))
        info["res_action_norm"] = float(np.linalg.norm(res))
        info["total_action_norm"] = float(np.linalg.norm(a)); info["residual_limit"] = self.residual_limit.copy()
        return obs, float(reward), terminated, truncated, info


def mild_curriculum_reward_shaper(phase, original_reward, obs, action, info):
    name = str(phase.get("name", "")); twist = float(np.clip(phase.get("twist_factor", 1.0), 0.0, 1.0))
    yaw_ref = float(info.get("yaw_ref", 0.0)); yaw_error = float(info.get("yaw_error", 0.0))
    z_asym = float(info.get("z_asym", 0.0))
    turn_gate = float(np.clip(abs(yaw_ref) / YAW_REF_MAX, 0.0, 1.0)); shaped = float(original_reward)
    if name == "basic_yaw":
        shaped += 0.015 * float(np.tanh(6.0 * np.linalg.norm(action)))
    else:
        shaped += (0.10 * twist) * turn_gate * float(np.clip(-yaw_error * z_asym, -0.20, +0.20))
    return float(np.clip(shaped, -15.0, +8.0))


class ProgressiveTwistWrapper(gym.Wrapper):
    def __init__(self, env, *, phase, twist_factor, reward_shaper=mild_curriculum_reward_shaper,
                 ramp_steps=0, start_twist_factor=None):
        super().__init__(env); self.phase = dict(phase)
        self.target_twist_factor = float(np.clip(twist_factor, 0.0, 1.0))
        self.start_twist_factor = float(
            self.target_twist_factor if start_twist_factor is None else np.clip(start_twist_factor, 0.0, 1.0))
        self.ramp_steps = int(max(0, ramp_steps)); self._ramp_t = 0; self.reward_shaper = reward_shaper
        self._apply_twist_enabled(self.target_twist_factor)

    def _effective_twist(self):
        if self.ramp_steps <= 0: return self.target_twist_factor
        frac = min(1.0, float(self._ramp_t) / float(self.ramp_steps))
        return float(self.start_twist_factor + frac * (self.target_twist_factor - self.start_twist_factor))

    def _apply_twist_enabled(self, tf):
        base = self.env.unwrapped
        if hasattr(base, "twist_enabled"): base.twist_enabled = bool(tf > 0.0)
        try: base._apply_twist_lock()
        except Exception: pass

    def set_phase(self, phase):
        self.phase = dict(phase)
        new_target = float(np.clip(self.phase.get("twist_factor", self.target_twist_factor), 0.0, 1.0))
        self.start_twist_factor = float(self._effective_twist())
        self.target_twist_factor = new_target; self._ramp_t = 0
        self._apply_twist_enabled(self.target_twist_factor)

    def reset(self, *, seed=None, options=None):
        self._ramp_t = 0; self._apply_twist_enabled(self.target_twist_factor)
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info); info["twist_factor"] = float(self._effective_twist())
        info["curriculum_phase_name"] = str(self.phase.get("name", "")); return obs, info

    def step(self, action):
        eff = self._effective_twist(); self._ramp_t += 1
        a = np.array(action, dtype=np.float32, copy=True)
        if eff <= 0.0: a[2] = 0.0; a[5] = 0.0
        else: a[2] *= float(eff); a[5] *= float(eff)
        obs, reward, terminated, truncated, info = self.env.step(a)
        info = dict(info); info["twist_factor"] = float(eff)
        info["curriculum_phase_name"] = str(self.phase.get("name", ""))
        if self.reward_shaper is not None:
            shaped = float(self.reward_shaper(self.phase, float(reward), obs, a, info))
            info["original_reward"] = float(reward); info["shaped_reward"] = float(shaped); reward = shaped
        return obs, float(reward), terminated, truncated, info
