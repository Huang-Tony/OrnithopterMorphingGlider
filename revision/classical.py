"""Classical baselines for revision R1, designed on the corrected plant (revision_runs/analysis/plant_identification.json).

Virtual morphing inputs (R = +y wing, L = -y wing; y is mirrored between wings):
    d_ds  differential sweep   dx_R = +d, dx_L = -d      -> yaw
    d_ss  symmetric sweep      dx_R = dx_L = +d          -> pitch
    d_dd  differential dihedral dz_R = +d, dz_L = -d     -> roll (+ yaw)
    d_sd  symmetric dihedral   dz_R = dz_L = +d          -> pitch
    d_dy  differential span    dy_R = dy_L = +d          -> roll (+ yaw)
    d_sy  symmetric span       dy_R = +d, dy_L = -d      -> pitch, lift
Controllers implement .reset() and .predict(obs) -> (action[6], None) like the SB3 wrapper.
"""

import json
import math
import os

import numpy as np

from common import RUNS_DIR
from morphing_glider.config import DT, DX_RANGE, DY_RANGE, DZ_RANGE
from morphing_glider.environment.observation import OBS_IDX

VIRT_NAMES = ["diff_sweep", "sym_sweep", "diff_dihedral", "sym_dihedral", "diff_span", "sym_span"]
LOW = np.array([DX_RANGE[0], DY_RANGE[0], DZ_RANGE[0]] * 2, dtype=float)
HIGH = np.array([DX_RANGE[1], DY_RANGE[1], DZ_RANGE[1]] * 2, dtype=float)


def virtual_to_tips(v):
    ds, ss, dd, sd, dy, sy = [float(x) for x in v]
    a = np.array([ss + ds, sy + dy, sd + dd, ss - ds, -sy + dy, sd - dd], dtype=float)
    return np.clip(a, LOW, HIGH)


def load_plant():
    with open(os.path.join(RUNS_DIR, "analysis", "plant_identification.json")) as f:
        P = json.load(f)
    B = np.array([P["control_effectiveness"][k]["dM_ddelta"] for k in VIRT_NAMES], dtype=float).T  # 3 x 6
    return P, B


def _att(obs):
    o = np.asarray(obs, dtype=float).reshape(-1)
    roll = math.atan2(o[OBS_IDX["sin_roll"]], o[OBS_IDX["cos_roll"]])
    pitch = math.atan2(o[OBS_IDX["sin_pitch"]], o[OBS_IDX["cos_pitch"]])
    return o, roll, pitch


class ZeroController:
    def reset(self): pass
    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return np.zeros(6, dtype=np.float32), state


class SISOYawController:
    """Yaw-rate loop acting on differential sweep only (the paper's PID / GS-PID / LQR / MPC family).
    kind: 'pid' (PI-D, pole placement), 'gspid' (airspeed-scheduled PID), 'lqr' (discrete LQR with
    integral action), 'mpc' (receding-horizon QP on the identified first-order model)."""

    def __init__(self, kind, *, scale=0.5, gain=1.0, plant=None):
        P = plant or load_plant()[0]
        ym = P["yaw_model"]
        self.kind = kind; self.scale = float(scale); self.gain = float(gain); self.dt = float(DT)
        self.a = ym["total_yaw_damping"] / ym["Izz"]                  # 1/s
        self.b = ym["N_delta_diff_sweep"] * self.scale / ym["Izz"]    # (rad/s^2) per unit command
        self.V0 = float(P["V0"])
        wn, zeta = 3.0 * self.gain, 0.8                               # closed-loop design point
        self.Kp = max(0.0, (2 * zeta * wn - self.a) / self.b); self.Ki = wn ** 2 / self.b; self.Kd = 0.02 * self.gain
        Ad = 1.0 - self.dt * self.a; Bd = self.dt * self.b
        if kind == "lqr":   # augmented state [e_r, integral e_r]
            A = np.array([[Ad, 0.0], [self.dt, 1.0]]); Bm = np.array([[Bd], [0.0]])
            Q = np.diag([1.0, 2.0 * self.gain ** 2]); R = np.array([[0.05]])
            Pm = np.eye(2)
            for _ in range(2000):
                K = np.linalg.solve(R + Bm.T @ Pm @ Bm, Bm.T @ Pm @ A)
                Pn = Q + A.T @ Pm @ (A - Bm @ K)
                if np.max(np.abs(Pn - Pm)) < 1e-10: Pm = Pn; break
                Pm = Pn
            self.K_lqr = np.linalg.solve(R + Bm.T @ Pm @ Bm, Bm.T @ Pm @ A).reshape(-1)
        self.Ad, self.Bd = Ad, Bd
        self.reset()

    def reset(self):
        self.i = 0.0; self.prev_e = 0.0

    def _u(self, e, r, rref, V):
        if self.kind in ("pid", "gspid"):
            Kp, Ki, Kd = self.Kp, self.Ki, self.Kd
            if self.kind == "gspid":   # gains scheduled with dynamic pressure (control power ~ V^2)
                s = (self.V0 / max(V, 5.0)) ** 2; Kp, Ki = Kp * s, Ki * s
            d = (e - self.prev_e) / self.dt; self.prev_e = e
            return Kp * e + Ki * self.i + Kd * d
        if self.kind == "lqr":
            return float(-self.K_lqr @ np.array([-e, -self.i]))
        if self.kind == "mpc":
            return self._mpc(r, rref)
        raise ValueError(self.kind)

    def _mpc(self, r, rref, N=10):
        """Offset-free MPC: state X=[e, z], e=r-r_ref, z=int e; e_{k+1}=Ad e_k+Bd u_k-(1-Ad) r_ref,
        z_{k+1}=z_k+dt e_k; min sum X'QX + R u^2 s.t. |u|<=1, solved as bounded least squares.
        The prediction matrices depend only on (Ad, Bd, gain) and are built once."""
        from scipy.optimize import lsq_linear
        if getattr(self, "_mpc_cache", None) is None:
            A = np.array([[self.Ad, 0.0], [self.dt, 1.0]]); Bv = np.array([self.Bd, 0.0])
            Apow = [np.linalg.matrix_power(A, k) for k in range(N + 1)]
            Gam = np.zeros((2 * N, N))
            for k in range(N):
                for j in range(k + 1):
                    Gam[2 * k:2 * k + 2, j] = Apow[k - j] @ Bv
            Qs = np.sqrt(np.array([1.0, 2.0 * self.gain ** 2])); W = np.tile(Qs, N)
            A_ls = np.vstack([W[:, None] * Gam, math.sqrt(0.05) * np.eye(N)])
            Phi = np.vstack([Apow[k + 1] for k in range(N)])                       # (2N x 2)
            Cs = np.vstack([sum(Apow[j] for j in range(k + 1)) for k in range(N)])   # accumulates the constant c
            self._mpc_cache = (A_ls, W, Phi, Cs)
        A_ls, W, Phi, Cs = self._mpc_cache
        c = np.array([-(1.0 - self.Ad) * rref, 0.0])
        off = Phi @ np.array([r - rref, -self.i]) + Cs @ c
        b = np.concatenate([-W * off, np.zeros(N)])
        return float(lsq_linear(A_ls, b, bounds=(-1.0, 1.0), method="bvls").x[0])

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        o = np.asarray(obs, dtype=float).reshape(-1)
        r = float(o[OBS_IDX["omega_r"]]); rref = float(o[OBS_IDX["yaw_ref"]]); V = float(o[OBS_IDX["speed"]])
        e = rref - r
        u = self._u(e, r, rref, V)
        u_sat = float(np.clip(u, -1.0, 1.0))
        if abs(u - u_sat) < 1e-9 or np.sign(e) != np.sign(u):   # conditional integration (anti-windup)
            self.i = float(np.clip(self.i + e * self.dt, -2.0, 2.0))
        return virtual_to_tips([u_sat * self.scale, 0, 0, 0, 0, 0]).astype(np.float32), state


class HeuristicV2:
    """The paper's 'virtual tendon' heuristic with channel signs corrected for the fixed plant:
    yaw = differential sweep (P on yaw-rate error), roll damping = TRUE differential span (restoring
    sign), pitch damping = symmetric sweep (restoring sign), optional dihedral feed-forward on yaw_ref."""

    def __init__(self, *, x_range=0.25, yaw_rate_max=0.6, z_ff_slope=0.0, roll_gain=0.15, roll_rate_gain=0.05,
                 pitch_gain=0.10, pitch_rate_gain=0.03, smooth=0.25, deadband=0.02):
        self.__dict__.update(dict(x_range=x_range, yaw_rate_max=yaw_rate_max, z_ff_slope=z_ff_slope,
                                  roll_gain=roll_gain, roll_rate_gain=roll_rate_gain, pitch_gain=pitch_gain,
                                  pitch_rate_gain=pitch_rate_gain, smooth=smooth, deadband=deadband))
        self.reset()

    def reset(self): self.prev = np.zeros(6)

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        o, roll, pitch = _att(obs)
        e = float(o[OBS_IDX["yaw_ref"]] - o[OBS_IDX["omega_r"]])
        if abs(e) < self.deadband: e = 0.0
        u = float(np.clip(e / self.yaw_rate_max, -1, 1))
        ds = self.x_range * u
        dd = self.z_ff_slope * float(o[OBS_IDX["yaw_ref"]])
        dy = -self.roll_gain * (math.sin(roll) + self.roll_rate_gain * float(o[OBS_IDX["omega_p"]]))
        ss = +self.pitch_gain * (math.sin(pitch) + self.pitch_rate_gain * float(o[OBS_IDX["omega_q"]]))
        cmd = virtual_to_tips([ds, ss, dd, 0.0, dy, 0.0])
        cmd = self.smooth * self.prev + (1 - self.smooth) * cmd; self.prev = cmd
        return cmd.astype(np.float32), state


class AllocationMultiLoop:
    """Attitude-stabilised multi-loop controller with model-based control allocation.

    Outer loops (desired body moments):
        M_x* = -I_xx (k_phi * phi + k_p * p)
        M_y* = -I_yy (k_theta * (theta - theta_ref) + k_q * q),  theta_ref = theta_trim + k_V (V_ref - V) (pitch +ve nose-down)
        M_z* =  I_zz (k_r * e_r + k_i * int e_r) + D_r * r_ref        (e_r = r_ref - r)
    Allocation: v* = argmin ||W (B v - M*)||^2 + lam ||v||^2  s.t. box bounds on the six virtual inputs,
    with B the identified 3x6 control-effectiveness matrix; v* is mapped to the six tip offsets."""

    def __init__(self, *, k_att=1.0, k_yaw=1.0, theta_ref=None, v_ref=15.0, k_v=0.04, lam=1e-3, plant=None):
        from scipy.optimize import lsq_linear
        self._lsq = lsq_linear
        P, B = (plant, load_plant()[1]) if plant is not None else load_plant()
        ph = P["phys"]; ym = P["yaw_model"]
        self.B = B; self.I = np.array([ph["Ixx"], ph["Iyy"], ph["Izz"]]); self.Dr = ym["total_yaw_damping"]
        wa = 6.0 * k_att; wy = 4.0 * k_yaw
        self.k_phi, self.k_p = wa ** 2, 2 * 0.8 * wa
        self.k_theta, self.k_q = (0.5 * wa) ** 2, 2 * 0.8 * 0.5 * wa
        self.k_r, self.k_i = 2 * 0.8 * wy, 0.5 * wy ** 2
        # airspeed-hold outer loop. NOTE: in this simulator's frame (x fwd, y to the R wing, z up) the Euler
        # pitch from quat_to_euler_xyz is POSITIVE NOSE-DOWN; the steady-glide attitude is +atan(D/L).
        self.theta_trim = math.atan(1.0 / float(P["trim"]["L_over_D"])) if theta_ref is None else float(theta_ref)
        self.v_ref = float(v_ref); self.k_v = float(k_v); self.dt = float(DT)
        self.W = np.diag([1.0, 0.5, 2.0]); self.lam = float(lam)
        self.lb = np.array([-0.35, -0.15, -0.10, -0.05, -0.15, -0.05])
        self.ub = -self.lb
        self.A_ls = np.vstack([self.W @ B, math.sqrt(self.lam) * np.eye(6)])
        self.reset()

    def reset(self): self.i = 0.0

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        o, roll, pitch = _att(obs)
        p, q, r = (float(o[OBS_IDX[k]]) for k in ("omega_p", "omega_q", "omega_r"))
        rref = float(o[OBS_IDX["yaw_ref"]]); e = rref - r
        self.i = float(np.clip(self.i + e * self.dt, -1.5, 1.5))
        V = float(o[OBS_IDX["speed"]])
        theta_ref = float(np.clip(self.theta_trim + self.k_v * (self.v_ref - V), -0.3, 0.6))   # slow -> nose down
        Mx = -self.I[0] * (self.k_phi * roll + self.k_p * p)
        My = -self.I[1] * (self.k_theta * (pitch - theta_ref) + self.k_q * q)
        Mz = self.I[2] * (self.k_r * e + self.k_i * self.i) + self.Dr * rref
        b = np.concatenate([self.W @ np.array([Mx, My, Mz]), np.zeros(6)])
        v = self._lsq(self.A_ls, b, bounds=(self.lb, self.ub), method="bvls").x
        return virtual_to_tips(v).astype(np.float32), state


def make_classical(name, **params):
    if name == "Zero": return ZeroController()
    if name in ("PID", "GS-PID", "LQR", "MPC"):
        return SISOYawController({"PID": "pid", "GS-PID": "gspid", "LQR": "lqr", "MPC": "mpc"}[name], **params)
    if name == "Heuristic": return HeuristicV2(**params)
    if name == "Alloc-ML": return AllocationMultiLoop(**params)
    raise ValueError(name)


# Tuning grids (identical budget: 9 configurations each, selected on held-out episodes)
TUNING_GRIDS = {
    "PID":       [dict(scale=s, gain=g) for s in (0.15, 0.3, 0.5) for g in (0.5, 1.0, 2.0)],
    "GS-PID":    [dict(scale=s, gain=g) for s in (0.15, 0.3, 0.5) for g in (0.5, 1.0, 2.0)],
    "LQR":       [dict(scale=s, gain=g) for s in (0.15, 0.3, 0.5) for g in (0.5, 1.0, 2.0)],
    "MPC":       [dict(scale=s, gain=g) for s in (0.15, 0.3, 0.5) for g in (0.5, 1.0, 2.0)],
    "Heuristic": [dict(x_range=x, roll_gain=k) for x in (0.15, 0.25, 0.45) for k in (0.08, 0.15, 0.3)],
    "Alloc-ML":  [dict(k_att=a, k_yaw=y) for a in (1.0, 2.0, 3.0) for y in (0.5, 1.0, 1.5)],
    "Zero":      [dict()],
}
