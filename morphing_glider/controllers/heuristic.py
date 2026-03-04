import numpy as np

from morphing_glider.environment.observation import OBS_IDX
from morphing_glider.config import DX_RANGE, DY_RANGE, DZ_RANGE


class VirtualTendonHeuristicController:
    """Bio-inspired heuristic controller with differential sweep, AI-learned
    z-twist feedforward, and roll/pitch damping.

    Yaw authority: differential x-sweep (both wings, opposite directions)
    plus differential z-twist with slope from machine teaching (R^2=0.91).
    Attitude stabilization: differential y for roll damping, symmetric x
    adjustment for pitch damping.
    """

    def __init__(self, *, yaw_rate_max=0.6, deadband=0.02, smooth=0.25,
                 x_range=0.25, z_ff_slope=-0.1408,
                 roll_gain=0.15, roll_rate_gain=0.05,
                 pitch_gain=0.10, pitch_rate_gain=0.03):
        self.yaw_rate_max = float(max(1e-6, yaw_rate_max))
        self.deadband = float(deadband)
        self.smooth = float(np.clip(smooth, 0.0, 0.98))
        self.x_range = float(x_range)
        self.z_ff_slope = float(z_ff_slope)
        self.roll_gain = float(roll_gain)
        self.roll_rate_gain = float(roll_rate_gain)
        self.pitch_gain = float(pitch_gain)
        self.pitch_rate_gain = float(pitch_rate_gain)
        self._prev_cmd = np.zeros(6, dtype=float)

    def reset(self):
        self._prev_cmd[:] = 0.0

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        obs = np.asarray(observation, dtype=float).reshape(-1)

        # --- Read observation ---
        yaw_ref = float(obs[OBS_IDX["yaw_ref"]])
        omega_r = float(obs[OBS_IDX["omega_r"]])
        sin_roll = float(obs[OBS_IDX["sin_roll"]])
        cos_roll = float(obs[OBS_IDX["cos_roll"]])
        sin_pitch = float(obs[OBS_IDX["sin_pitch"]])
        cos_pitch = float(obs[OBS_IDX["cos_pitch"]])
        omega_p = float(obs[OBS_IDX["omega_p"]])
        omega_q = float(obs[OBS_IDX["omega_q"]])

        # --- 1. YAW: Differential x-sweep + AI z-twist feedforward ---
        e_yaw = yaw_ref - omega_r
        if abs(e_yaw) < self.deadband:
            e_yaw = 0.0
        u_yaw = float(np.clip(e_yaw / self.yaw_rate_max, -1.0, 1.0))

        # Differential x-sweep: both wings, opposite directions
        dx_yaw_R = +self.x_range * u_yaw
        dx_yaw_L = -self.x_range * u_yaw

        # AI-discovered z-twist feedforward (slope = -0.1408 m/(rad/s))
        dz_R = float(np.clip(self.z_ff_slope * yaw_ref, DZ_RANGE[0], DZ_RANGE[1]))
        dz_L = float(np.clip(-self.z_ff_slope * yaw_ref, DZ_RANGE[0], DZ_RANGE[1]))

        # --- 2. ROLL DAMPING: Differential y to counteract roll ---
        roll_err = sin_roll  # sin(roll) ~ roll for small angles
        dy_roll = -self.roll_gain * (roll_err + self.roll_rate_gain * omega_p)
        dy_R = +dy_roll
        dy_L = -dy_roll

        # --- 3. PITCH DAMPING: Symmetric x adjustment ---
        pitch_err = sin_pitch  # sin(pitch) ~ pitch for small angles
        dx_pitch = -self.pitch_gain * (pitch_err + self.pitch_rate_gain * omega_q)

        # --- Combine ---
        dx_R = dx_yaw_R + dx_pitch
        dx_L = dx_yaw_L + dx_pitch

        cmd = np.array([dx_R, dy_R, dz_R, dx_L, dy_L, dz_L], dtype=float)

        # Exponential smoothing
        cmd = self.smooth * self._prev_cmd + (1.0 - self.smooth) * cmd
        self._prev_cmd = cmd.copy()

        # Clip to action bounds
        cmd = np.clip(cmd,
                      [DX_RANGE[0], DY_RANGE[0], DZ_RANGE[0]] * 2,
                      [DX_RANGE[1], DY_RANGE[1], DZ_RANGE[1]] * 2)
        return cmd.astype(np.float32), state
