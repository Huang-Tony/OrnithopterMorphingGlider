import math
from typing import List, Optional, Tuple

import numpy as np

from morphing_glider.config import DT
from morphing_glider.environment.observation import OBS_IDX


class PIDYawController:
    """PID yaw-rate tracker for morphing glider."""
    def __init__(self, Kp=0.8, Ki=0.05, Kd=0.02, dt=DT, action_scale=0.15, integral_limit=1.0):
        self.Kp = float(Kp); self.Ki = float(Ki); self.Kd = float(Kd)
        self.dt = float(dt); self.action_scale = float(action_scale); self.integral_limit = float(integral_limit)
        self._integral = 0.0; self._prev_error = 0.0

    def reset(self): self._integral = 0.0; self._prev_error = 0.0

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        if isinstance(obs, dict):
            r = float(obs.get("yaw_rate", 0.0)); r_ref = float(obs.get("yaw_ref", 0.0))
        else:
            o = np.asarray(obs, dtype=float).reshape(-1)
            r = float(o[OBS_IDX["omega_r"]]); r_ref = float(o[OBS_IDX["yaw_ref"]])
        error = r_ref - r
        self._integral = float(np.clip(self._integral + error * self.dt, -self.integral_limit, self.integral_limit))
        derivative = (error - self._prev_error) / max(self.dt, 1e-9); self._prev_error = float(error)
        u = float(np.clip(self.Kp * error + self.Ki * self._integral + self.Kd * derivative, -1.0, 1.0))
        da = u * self.action_scale
        return np.array([da, 0.0, 0.0, -da, 0.0, 0.0], dtype=np.float32), state

    def tune_from_aero(self, Izz, K_mz_per_dx):
        self.Kp = float(np.sqrt(2.0 * float(Izz) / max(float(K_mz_per_dx), 1e-6)))
        print(f"  [PID] Auto-tuned Kp={self.Kp:.4f}")
    auto_tune_from_aero = tune_from_aero


class GainScheduledPIDYawController:
    """PID yaw-rate controller with gain scheduling over airspeed."""
    def __init__(self, schedule=None, dt=DT, action_scale=0.15, integral_limit=1.0):
        self.dt = float(dt); self.action_scale = float(action_scale)
        self.integral_limit = float(integral_limit)
        if schedule is None:
            schedule = [(10.0, 1.2, 0.08, 0.03), (15.0, 0.8, 0.05, 0.02), (20.0, 0.5, 0.03, 0.01)]
        self.schedule = sorted(schedule, key=lambda t: t[0])
        self._integral = 0.0; self._prev_error = 0.0
        print(f"  [GS-PID] Schedule: {self.schedule}")

    def auto_tune_from_aero(self, Izz, K_mz_per_dx):
        kp_nom = float(np.sqrt(2.0 * float(Izz) / max(float(K_mz_per_dx), 1e-6)))
        self.schedule = [(10.0, kp_nom * 1.5, 0.08, 0.03),
                         (15.0, kp_nom, 0.05, 0.02),
                         (20.0, kp_nom * 0.6, 0.03, 0.01)]
        print(f"  [GS-PID] Auto-tuned schedule: {self.schedule}")

    def reset(self): self._integral = 0.0; self._prev_error = 0.0

    def _interpolate_gains(self, airspeed):
        V = float(airspeed)
        if V <= self.schedule[0][0]: return self.schedule[0][1], self.schedule[0][2], self.schedule[0][3]
        if V >= self.schedule[-1][0]: return self.schedule[-1][1], self.schedule[-1][2], self.schedule[-1][3]
        for i in range(len(self.schedule) - 1):
            v0, kp0, ki0, kd0 = self.schedule[i]
            v1, kp1, ki1, kd1 = self.schedule[i + 1]
            if v0 <= V <= v1:
                t = (V - v0) / max(v1 - v0, 1e-9)
                return kp0 + t * (kp1 - kp0), ki0 + t * (ki1 - ki0), kd0 + t * (kd1 - kd0)
        return self.schedule[0][1], self.schedule[0][2], self.schedule[0][3]

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        o = np.asarray(obs, dtype=float).reshape(-1)
        r = float(o[OBS_IDX["omega_r"]]); r_ref = float(o[OBS_IDX["yaw_ref"]])
        airspeed = float(o[OBS_IDX["speed"]])
        Kp, Ki, Kd = self._interpolate_gains(airspeed)
        error = r_ref - r
        self._integral = float(np.clip(self._integral + error * self.dt, -self.integral_limit, self.integral_limit))
        derivative = (error - self._prev_error) / max(self.dt, 1e-9); self._prev_error = float(error)
        u = float(np.clip(Kp * error + Ki * self._integral + Kd * derivative, -1.0, 1.0))
        da = u * self.action_scale
        return np.array([da, 0.0, 0.0, -da, 0.0, 0.0], dtype=np.float32), state
