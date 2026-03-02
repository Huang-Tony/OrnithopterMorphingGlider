import numpy as np

from morphing_glider.environment.observation import OBS_IDX
from morphing_glider.config import DX_RANGE, DY_RANGE, DZ_RANGE


class VirtualTendonHeuristicController:
    def __init__(self, *, yaw_rate_max=0.6, deadband=0.02, smooth=0.25,
                 x_bias=0.02, x_range=0.30, z_range=0.12, y_range=0.10, unload_retract=0.06):
        self.yaw_rate_max = float(max(1e-6, yaw_rate_max)); self.deadband = float(deadband)
        self.smooth = float(np.clip(smooth, 0.0, 0.98))
        self.x_bias = float(x_bias); self.x_range = float(x_range)
        self.z_range = float(z_range); self.y_range = float(y_range); self.unload_retract = float(unload_retract)
        self._prev_cmd = np.zeros(6, dtype=float)

    def reset(self): self._prev_cmd[:] = 0.0

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        obs = np.asarray(observation, dtype=float).reshape(-1)
        e = float(obs[OBS_IDX["yaw_ref"]]) - float(obs[OBS_IDX["omega_r"]])
        if abs(e) < self.deadband: e = 0.0
        u = float(np.clip(e / self.yaw_rate_max, -1.0, 1.0)); mag = abs(u)
        x_load = self.x_bias + self.x_range * mag; z_load = self.z_range * mag; y_load = self.y_range * mag
        cmd = np.zeros(6)
        if u > 0: cmd[0] = +x_load; cmd[1] = +y_load; cmd[2] = +z_load; cmd[4] = +self.unload_retract * mag
        elif u < 0: cmd[3] = +x_load; cmd[4] = -y_load; cmd[5] = +z_load; cmd[1] = -self.unload_retract * mag
        cmd = self.smooth * self._prev_cmd + (1.0 - self.smooth) * cmd
        self._prev_cmd = cmd
        cmd = np.clip(cmd, [DX_RANGE[0], DY_RANGE[0], DZ_RANGE[0]] * 2,
                      [DX_RANGE[1], DY_RANGE[1], DZ_RANGE[1]] * 2)
        return cmd.astype(np.float32), state
