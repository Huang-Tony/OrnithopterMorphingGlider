import numpy as np

try:
    from scipy.optimize import minimize as scipy_minimize
except Exception:
    scipy_minimize = None

from morphing_glider.config import DT, MPC_N_HORIZON, MPC_Q_R, MPC_R_U
from morphing_glider.environment.observation import OBS_IDX
from morphing_glider.controllers.pid import PIDYawController


class LinearMPCYawController:
    """Receding-horizon MPC controller for yaw rate tracking."""
    def __init__(self, Izz=0.120, K_mz_per_dx=2.128, d_yaw=0.35,
                 N_horizon=MPC_N_HORIZON, Q_r=MPC_Q_R, R_u=MPC_R_U,
                 dt=DT, action_scale=0.15):
        self.Izz = float(Izz); self.K_mz = float(K_mz_per_dx); self.d_yaw = float(d_yaw)
        self.N = int(N_horizon); self.Q_r = float(Q_r); self.R_u = float(R_u)
        self.dt = float(dt); self.action_scale = float(action_scale)
        self.A_d = 1.0 - self.dt * self.d_yaw / max(self.Izz, 1e-9)
        self.B_d = self.dt * 2.0 * self.action_scale * self.K_mz / max(self.Izz, 1e-9)
        self._fallback_pid = PIDYawController(dt=dt, action_scale=action_scale)
        print(f"  [MPC] N={self.N}, A_d={self.A_d:.4f}, B_d={self.B_d:.4f}")

    def reset(self): self._fallback_pid.reset()

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        if isinstance(obs, dict):
            r = float(obs.get("yaw_rate", 0.0)); r_ref = float(obs.get("yaw_ref", 0.0))
        else:
            o = np.asarray(obs, dtype=float).reshape(-1)
            r = float(o[OBS_IDX["omega_r"]]); r_ref = float(o[OBS_IDX["yaw_ref"]])
        if scipy_minimize is None:
            return self._fallback_pid.predict(obs, state, episode_start, deterministic)
        x0 = r - r_ref; N = self.N; Ad = self.A_d; Bd = self.B_d; Qr = self.Q_r; Ru = self.R_u

        def cost(u_vec):
            x = x0; J = 0.0
            for k in range(N):
                J += Qr * x * x + Ru * u_vec[k] * u_vec[k]
                x = Ad * x + Bd * u_vec[k]
            J += Qr * x * x
            return J

        u0 = np.zeros(N); bounds = [(-1.0, 1.0)] * N
        try:
            res = scipy_minimize(cost, u0, method='SLSQP', bounds=bounds,
                                 options={'maxiter': 50, 'ftol': 1e-6})
            u_opt = float(np.clip(res.x[0], -1.0, 1.0))
        except Exception:
            return self._fallback_pid.predict(obs, state, episode_start, deterministic)
        da = u_opt * self.action_scale
        return np.array([da, 0.0, 0.0, -da, 0.0, 0.0], dtype=np.float32), state
