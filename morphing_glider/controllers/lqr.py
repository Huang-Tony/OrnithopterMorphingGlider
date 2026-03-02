import math
import numpy as np

from morphing_glider.config import DT
from morphing_glider.environment.observation import OBS_IDX


class LQRYawController:
    """LQR yaw-rate tracker using linearized 1-state model."""
    def __init__(self, Izz=0.120, K_mz_per_dx=2.128, Q=1.0, R=0.1, dt=DT, action_scale=0.15):
        self.Izz = float(Izz); self.K_mz_per_dx = float(K_mz_per_dx)
        self.Q = float(Q); self.R = float(R)
        self.dt = float(dt); self.action_scale = float(action_scale)
        self.Bd = float(self.dt * (2.0 * self.action_scale * self.K_mz_per_dx / max(self.Izz, 1e-9)))
        self.K = self._compute_discrete_lqr_gain(self.Bd, self.Q, self.R)
        print(f"  [LQR] K={self.K:.4f}, Bd={self.Bd:.4f}")

    @staticmethod
    def _compute_discrete_lqr_gain(Bd, Q, R):
        Bd = float(Bd); Q = float(max(0.0, Q)); R = float(max(1e-12, R))
        if abs(Bd) < 1e-12: return 0.0
        disc = Q * Q + 4.0 * Q * R / (Bd * Bd)
        P = 0.5 * (Q + math.sqrt(max(0.0, disc)))
        return float((Bd * P) / (R + (Bd * Bd) * P + 1e-12))

    def reset(self): pass

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        if isinstance(obs, dict):
            r = float(obs.get("yaw_rate", 0.0)); r_ref = float(obs.get("yaw_ref", 0.0))
        else:
            o = np.asarray(obs, dtype=float).reshape(-1)
            r = float(o[OBS_IDX["omega_r"]]); r_ref = float(o[OBS_IDX["yaw_ref"]])
        u_norm = float(np.clip(-self.K * (r - r_ref), -1.0, 1.0))
        da = u_norm * self.action_scale
        return np.array([da, 0.0, 0.0, -da, 0.0, 0.0], dtype=np.float32), state
