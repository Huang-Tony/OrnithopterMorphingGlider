"""MorphingGliderEnv6DOF — 41D obs, 6D action, quaternion-based 6DOF dynamics."""

import math
from typing import Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from morphing_glider.config import (
    DT, L_FIXED, DX_RANGE, DY_RANGE, DZ_RANGE,
    DEFAULT_YAW_TARGETS, HOLD_RANGE_STEPS,
    FAST_DEV_RUN,
)
from morphing_glider.environment.observation import OBS_IDX, OBS_DIM
from morphing_glider.environment.reward import RewardComputer
from morphing_glider.physics.bezier_spar import RealTimeBezierSpar
from morphing_glider.physics.aero_proxy import AeroProxy3D
from morphing_glider.physics.domain_randomizer import DomainRandomizer, NOMINAL_PHYS
from morphing_glider.utils.quaternion import (
    quat_normalize, quat_mul, quat_to_rotmat_body_to_world,
    quat_integrate_body_rates, quat_to_euler_xyz,
)

YAW_REF_MAX = float(max(1e-6, np.max(np.abs(DEFAULT_YAW_TARGETS))))


class MorphingGliderEnv6DOF(gym.Env):
    """41D obs; 6D action (tip offsets). Rotation + translation; quasi-steady aero; DR."""
    metadata = {"render_modes": []}
    _E_SUM_MAX_CACHE: Optional[float] = None

    def __init__(self, *, max_steps=200, twist_enabled=True, include_omega_cross=True,
                 yaw_targets=DEFAULT_YAW_TARGETS, hold_range_steps=HOLD_RANGE_STEPS,
                 num_aero_panels=12, domain_rand_scale=0.0, domain_rand_enabled=True,
                 actuator_tau=0.07, start_altitude=200.0, speed_min_terminate=6.0,
                 roll_pitch_limit_deg=70.0, terminal_fail_penalty=12.0,
                 coupling_scale=1.0, stability_weight=0.03,
                 sensor_noise_scale: float = 1.0,
                 reward_computer: Optional[RewardComputer] = None,
                 seed=None):
        super().__init__()
        self.dt = float(DT); self.max_steps = int(max_steps)
        self.twist_enabled = bool(twist_enabled); self.include_omega_cross = bool(include_omega_cross)
        self.yaw_targets = list(map(float, yaw_targets))
        self.hold_min = int(hold_range_steps[0]); self.hold_max = int(hold_range_steps[1])
        self.act_tau = float(max(1e-3, actuator_tau))
        self.start_altitude = float(start_altitude)
        self.speed_min_terminate = float(speed_min_terminate)
        self.roll_pitch_limit = math.radians(float(roll_pitch_limit_deg))
        self.terminal_fail_penalty = float(max(0.0, terminal_fail_penalty))
        self.base_terminal_penalty = float(self.terminal_fail_penalty)
        self.coupling_scale = float(np.clip(coupling_scale, 0.0, 1.0))
        self.stability_weight = float(max(0.0, stability_weight))
        self.sensor_noise_scale = float(max(0.0, sensor_noise_scale))
        self.reward_computer = reward_computer if reward_computer is not None else RewardComputer()
        self.randomizer = DomainRandomizer(enabled=domain_rand_enabled, scale=float(domain_rand_scale))
        self.phys = dict(NOMINAL_PHYS)
        low = np.array([DX_RANGE[0], DY_RANGE[0], DZ_RANGE[0], DX_RANGE[0], DY_RANGE[0], DZ_RANGE[0]], dtype=np.float32)
        high = np.array([DX_RANGE[1], DY_RANGE[1], DZ_RANGE[1], DX_RANGE[1], DY_RANGE[1], DZ_RANGE[1]], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.aero = AeroProxy3D(num_panels=int(num_aero_panels), include_omega_cross=bool(include_omega_cross))
        self.spar_R = RealTimeBezierSpar([0, 0, 0], [0, +L_FIXED, 0], [0, +L_FIXED * 0.33, 0], [0, +L_FIXED * 0.66, 0])
        self.spar_L = RealTimeBezierSpar([0, 0, 0], [0, -L_FIXED, 0], [0, -L_FIXED * 0.33, 0], [0, -L_FIXED * 0.66, 0])
        self._e_sum_max = float(self._estimate_max_structural_energy())
        self.current_step = 0
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.omega = np.zeros(3, dtype=float)
        self.pos_world = np.zeros(3, dtype=float)
        self.vel_world = np.zeros(3, dtype=float)
        self.yaw_ref = 0.0; self.yaw_ref_prev = 0.0; self.hold_count = self.hold_min
        self.gust = np.zeros(3, dtype=float)
        self.p3_R = np.array([0.0, +L_FIXED, 0.0]); self.p3_L = np.array([0.0, -L_FIXED, 0.0])
        self.p3_cmd_R = self.p3_R.copy(); self.p3_cmd_L = self.p3_L.copy()
        self._prev_action = np.zeros(6, dtype=float)
        self.reset(seed=seed)

    @classmethod
    def _compute_struct_energy_sum_for_tips(cls, *, p3_R, p3_L, lock_z=False,
                                             max_total_iters=120, chunk_iters=16, tol_len=1e-3):
        p3_R = np.asarray(p3_R, dtype=float).reshape(3)
        p3_L = np.asarray(p3_L, dtype=float).reshape(3)
        if lock_z:
            p3_R = p3_R.copy(); p3_L = p3_L.copy(); p3_R[2] = 0.0; p3_L[2] = 0.0
        sR = RealTimeBezierSpar([0, 0, 0], p3_R, 0.33 * p3_R, 0.66 * p3_R)
        sL = RealTimeBezierSpar([0, 0, 0], p3_L, 0.33 * p3_L, 0.66 * p3_L)
        sR.lock_z = bool(lock_z); sL.lock_z = bool(lock_z)
        sR.solve_to_convergence(max_total_iters=int(max_total_iters), chunk_iters=int(chunk_iters), tol_len=float(tol_len))
        sL.solve_to_convergence(max_total_iters=int(max_total_iters), chunk_iters=int(chunk_iters), tol_len=float(tol_len))
        _, eR = sR.length_and_energy(); _, eL = sL.length_and_energy()
        return float(eR + eL)

    @classmethod
    def get_e_sum_max_cached(cls):
        if cls._E_SUM_MAX_CACHE is not None:
            return float(cls._E_SUM_MAX_CACHE)
        fallback = 5.0
        try:
            dx = float(DX_RANGE[1]); dz = float(DZ_RANGE[1])
            yR = float(+L_FIXED + DY_RANGE[1]); yL = float(-L_FIXED + DY_RANGE[0])
            p3_R = np.array([dx, yR, dz]); p3_L = np.array([dx, yL, dz])
            e_sum_max = cls._compute_struct_energy_sum_for_tips(
                p3_R=p3_R, p3_L=p3_L,
                max_total_iters=80 if FAST_DEV_RUN else 120, chunk_iters=16, tol_len=1e-3)
            if (not np.isfinite(e_sum_max)) or (e_sum_max <= 1e-9):
                raise ValueError(f"bad e_sum_max={e_sum_max}")
        except Exception as e:
            print(f"[StructNorm] WARNING: Could not estimate e_sum_max ({e!r}); using fallback={fallback}.")
            e_sum_max = float(fallback)
        cls._E_SUM_MAX_CACHE = float(e_sum_max)
        return float(cls._E_SUM_MAX_CACHE)

    def _estimate_max_structural_energy(self):
        return float(self.get_e_sum_max_cached())

    def set_roll_pitch_limit_deg(self, deg):
        self.roll_pitch_limit = math.radians(float(deg)); return float(self.roll_pitch_limit)

    def set_coupling_scale(self, scale):
        self.coupling_scale = float(np.clip(scale, 0.0, 1.0)); return float(self.coupling_scale)

    def set_stability_weight(self, w):
        self.stability_weight = float(max(0.0, w)); return float(self.stability_weight)

    def _compute_terminal_penalty(self):
        sr = float(self.current_step) / max(1.0, float(self.max_steps))
        sr = float(np.clip(sr, 0.0, 1.0))
        lm = 1.0 + 3.0 * max(0.0, 1.0 - sr)
        base_penalty = float(self.base_terminal_penalty) * float(lm)
        remaining_steps = max(0, int(self.max_steps) - int(self.current_step))
        lost_potential = float(remaining_steps) * float(self.reward_computer.survival_bonus)
        total_penalty = base_penalty + lost_potential
        return float(total_penalty), float(sr), float(lm)

    def _apply_twist_lock(self):
        lock = not self.twist_enabled
        self.spar_R.lock_z = lock; self.spar_L.lock_z = lock
        if lock:
            for a in [self.p3_R, self.p3_L, self.p3_cmd_R, self.p3_cmd_L]:
                a[2] = 0.0
            self.spar_R.p3[2] = 0.0; self.spar_L.p3[2] = 0.0
            self.spar_R.p1[2] = 0.0; self.spar_R.p2[2] = 0.0
            self.spar_L.p1[2] = 0.0; self.spar_L.p2[2] = 0.0

    def _sample_new_yaw_ref(self):
        choices = [v for v in self.yaw_targets if abs(v - self.yaw_ref) > 1e-9]
        if not choices:
            return
        self.yaw_ref_prev = float(self.yaw_ref)
        self.yaw_ref = float(self.np_random.choice(choices))
        self.hold_count = int(self.np_random.integers(self.hold_min, self.hold_max + 1))

    def _update_gust(self):
        tau = float(max(1e-3, self.phys.get("gust_tau", 0.8))); dt = float(self.dt)
        alpha_g = math.exp(-dt / tau)
        sig = np.array([max(0.0, self.phys.get(f"gust_sigma_{c}", 0.0)) for c in "xyz"])
        noise = self.np_random.normal(0.0, 1.0, size=(3,))
        self.gust = alpha_g * self.gust + math.sqrt(max(0.0, 1.0 - alpha_g ** 2)) * sig * noise

    def _wind_world(self):
        return np.array([self.phys.get("wind_x", 0.0), self.phys.get("wind_y", 0.0),
                         self.phys.get("wind_z", 0.0)], dtype=float) + self.gust

    def _get_obs(self):
        roll, pitch, yaw = quat_to_euler_xyz(self.q)
        an = float(self.phys.get("imu_angle_noise", 0.0)); on = float(self.phys.get("imu_omega_noise", 0.0))
        asn = float(self.phys.get("airspeed_noise", 0.0)); aln = float(self.phys.get("alt_noise", 0.0))
        roll_m = roll + float(self.np_random.normal(0.0, an))
        pitch_m = pitch + float(self.np_random.normal(0.0, an))
        yaw_m = yaw + float(self.np_random.normal(0.0, an))
        omega_m = self.omega + self.np_random.normal(0.0, on, size=(3,))
        R_bw = quat_to_rotmat_body_to_world(self.q)
        v_rel_body = R_bw.T @ (self.vel_world - self._wind_world())
        v_rel_body_m = v_rel_body + self.np_random.normal(0.0, asn, size=(3,))
        speed_m = float(np.linalg.norm(v_rel_body_m)) + 1e-9
        alt_m = float(self.pos_world[2] + self.np_random.normal(0.0, aln))
        vz_m = float(self.vel_world[2] + self.np_random.normal(0.0, aln * 0.25))
        obs = np.zeros(OBS_DIM, dtype=float)
        obs[0] = math.sin(roll_m); obs[1] = math.cos(roll_m)
        obs[2] = math.sin(pitch_m); obs[3] = math.cos(pitch_m)
        obs[4] = math.sin(yaw_m); obs[5] = math.cos(yaw_m)
        obs[6:9] = omega_m; obs[9:12] = v_rel_body_m; obs[12] = speed_m
        obs[13] = alt_m; obs[14] = vz_m; obs[15] = float(self.yaw_ref); obs[16] = float(self.yaw_ref_prev)
        obs[17:20] = self.p3_R; obs[20:23] = self.p3_L
        obs[23:26] = self.p3_cmd_R; obs[26:29] = self.p3_cmd_L
        obs[29:32] = self.spar_R.p1; obs[32:35] = self.spar_R.p2
        obs[35:38] = self.spar_L.p1; obs[38:41] = self.spar_L.p2
        np.clip(obs, -1e6, 1e6, out=obs)
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.phys = self.randomizer.sample(self.np_random)
        for k in ["imu_omega_noise", "imu_angle_noise", "airspeed_noise", "alt_noise"]:
            self.phys[k] = float(self.phys[k] * self.sensor_noise_scale)
        self.current_step = 0
        self.q[:] = [1.0, 0.0, 0.0, 0.0]; self.omega[:] = 0.0
        self.pos_world[:] = [0.0, 0.0, float(self.start_altitude)]
        V0 = float(self.phys.get("V0", NOMINAL_PHYS["V0"]))
        self.vel_world[:] = [V0, 0.0, 0.0]; self.gust[:] = 0.0
        self.yaw_ref = float(self.np_random.choice(self.yaw_targets))
        self.yaw_ref_prev = float(self.yaw_ref)
        self.hold_count = int(self.np_random.integers(self.hold_min, self.hold_max + 1))
        self.p3_R[:] = [0.0, +L_FIXED, 0.0]; self.p3_L[:] = [0.0, -L_FIXED, 0.0]
        self.p3_cmd_R[:] = self.p3_R; self.p3_cmd_L[:] = self.p3_L
        self.spar_R.p3 = self.p3_R.copy(); self.spar_L.p3 = self.p3_L.copy()
        self.spar_R.p1 = np.array([0, +L_FIXED * 0.33, 0.0]); self.spar_R.p2 = np.array([0, +L_FIXED * 0.66, 0.0])
        self.spar_L.p1 = np.array([0, -L_FIXED * 0.33, 0.0]); self.spar_L.p2 = np.array([0, -L_FIXED * 0.66, 0.0])
        self._prev_action[:] = 0.0
        self._apply_twist_lock(); self.spar_R.solve_shape(); self.spar_L.solve_shape(); self._apply_twist_lock()
        info = {"yaw_ref": float(self.yaw_ref), "twist_enabled": bool(self.twist_enabled),
                "domain_rand_scale": float(self.randomizer.scale), "phys": dict(self.phys),
                "coupling_scale": float(self.coupling_scale), "stability_weight": float(self.stability_weight),
                "roll_pitch_limit_deg": float(math.degrees(self.roll_pitch_limit)), "e_sum_max": float(self._e_sum_max)}
        return self._get_obs(), info

    def step(self, action):
        self.current_step += 1; self.hold_count -= 1
        if self.hold_count <= 0:
            self._sample_new_yaw_ref()
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), self.action_space.low, self.action_space.high)
        if not self.twist_enabled:
            a[2] = 0.0; a[5] = 0.0
        self.p3_cmd_R = np.array([a[0], +L_FIXED + a[1], a[2]])
        self.p3_cmd_L = np.array([a[3], -L_FIXED + a[4], a[5]])
        alpha_act = 1.0 - math.exp(-self.dt / self.act_tau)
        self.p3_R += alpha_act * (self.p3_cmd_R - self.p3_R)
        self.p3_L += alpha_act * (self.p3_cmd_L - self.p3_L)
        self.p3_R[0] = np.clip(self.p3_R[0], DX_RANGE[0], DX_RANGE[1])
        self.p3_R[1] = np.clip(self.p3_R[1], L_FIXED + DY_RANGE[0], L_FIXED + DY_RANGE[1])
        self.p3_R[2] = np.clip(self.p3_R[2], DZ_RANGE[0], DZ_RANGE[1])
        self.p3_L[0] = np.clip(self.p3_L[0], DX_RANGE[0], DX_RANGE[1])
        self.p3_L[1] = np.clip(self.p3_L[1], -L_FIXED + DY_RANGE[0], -L_FIXED + DY_RANGE[1])
        self.p3_L[2] = np.clip(self.p3_L[2], DZ_RANGE[0], DZ_RANGE[1])
        if not self.twist_enabled:
            self.p3_R[2] = 0.0; self.p3_L[2] = 0.0; self.p3_cmd_R[2] = 0.0; self.p3_cmd_L[2] = 0.0
        self.spar_R.p3 = self.p3_R.copy(); self.spar_L.p3 = self.p3_L.copy()
        self._apply_twist_lock(); self.spar_R.solve_shape(); self.spar_L.solve_shape(); self._apply_twist_lock()
        self._update_gust()
        R_bw = quat_to_rotmat_body_to_world(self.q)
        v_rel_world = self.vel_world - self._wind_world(); v_rel_body = R_bw.T @ v_rel_world
        F_R, M_R, d_R = self.aero.calculate_forces(self.spar_R, v_rel_body=v_rel_body, omega_body=self.omega, phys=self.phys)
        F_L, M_L, d_L = self.aero.calculate_forces(self.spar_L, v_rel_body=v_rel_body, omega_body=self.omega, phys=self.phys)
        F_body = F_R + F_L; M_body = M_R + M_L
        cs = float(self.coupling_scale)
        M_used = np.array([M_body[0] * cs, M_body[1] * cs, M_body[2]], dtype=float)
        I = np.array([float(self.phys["Ixx"]), float(self.phys["Iyy"]), float(self.phys["Izz"])])
        D = np.array([float(self.phys["d_roll"]), float(self.phys["d_pitch"]), float(self.phys["d_yaw"])])
        omega_dot = (M_used - np.cross(self.omega, I * self.omega) - D * self.omega) / (I + 1e-9)
        self.omega = np.clip(self.omega + omega_dot * self.dt, -8.0, +8.0)
        self.q = quat_integrate_body_rates(self.q, self.omega, self.dt)
        m = float(self.phys.get("mass", 0.5)); g = float(self.phys.get("g", 9.81))
        F_world = R_bw @ F_body; gravity = np.array([0.0, 0.0, -m * g])
        self.vel_world += (F_world + gravity) / max(1e-9, m) * self.dt
        self.vel_world = np.clip(self.vel_world, -100.0, 100.0)
        self.pos_world += self.vel_world * self.dt
        roll, pitch, yaw = quat_to_euler_xyz(self.q)
        yaw_rate = float(self.omega[2]); speed = float(np.linalg.norm(v_rel_world))
        altitude = float(self.pos_world[2]); vz_world = float(self.vel_world[2])
        power_loss = float(max(0.0, -float(np.dot(F_body, v_rel_body))))
        yaw_error = yaw_rate - float(self.yaw_ref)
        zR = float(self.p3_R[2]); zL = float(self.p3_L[2])
        z_asym = 0.5 * (zR - zL); z_sym = 0.5 * (zR + zL)
        _, e_R = self.spar_R.length_and_energy(); _, e_L = self.spar_L.length_and_energy()
        e_sum_norm = float(e_R + e_L) / max(float(self._e_sum_max), 1e-3)
        power_norm = float(power_loss / max(1e-6, m * g * max(1.0, float(self.phys.get("V0", 15.0)))))
        omega_clipped = np.clip(self.omega, -8.0, +8.0)

        reward, breakdown = self.reward_computer.compute(
            yaw_error=yaw_error, roll=roll, pitch=pitch,
            omega_p_clipped=float(omega_clipped[0]), omega_q_clipped=float(omega_clipped[1]),
            action=a, prev_action=self._prev_action,
            power_norm=power_norm, e_sum_norm=e_sum_norm, z_sym=z_sym,
            stability_weight=self.stability_weight,
            roll_pitch_limit=float(self.roll_pitch_limit))

        self._prev_action = a.copy()
        truncated = (self.current_step >= self.max_steps)
        reason = None
        if abs(roll) > self.roll_pitch_limit or abs(pitch) > self.roll_pitch_limit:
            reason = "attitude_limit"
        if speed < self.speed_min_terminate and reason is None:
            reason = "stall"
        if altitude <= 0.0 and reason is None:
            reason = "ground"
        if not np.isfinite(reward) and reason is None:
            reason = "nan"
        terminated = reason is not None
        terminal_penalty = 0.0
        survival_ratio = float(self.current_step) / max(1.0, float(self.max_steps))
        penalty_mult = 1.0
        if terminated and not truncated:
            terminal_penalty, survival_ratio, penalty_mult = self._compute_terminal_penalty()
            reward = float(reward - terminal_penalty)

        obs = self._get_obs()
        if reason == "nan" or not np.isfinite(obs).all():
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            reward = self.reward_computer.clip_min
            terminated = True

        info = {
            "yaw_rate": yaw_rate, "yaw_ref": float(self.yaw_ref), "yaw_error": yaw_error,
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "omega_p": float(self.omega[0]), "omega_q": float(self.omega[1]), "omega_r": float(self.omega[2]),
            "moment_x": float(M_used[0]), "moment_y": float(M_used[1]), "moment_z": float(M_used[2]),
            "drag_R": float(d_R["total_drag_force"]), "drag_L": float(d_L["total_drag_force"]),
            "power_loss_R": float(d_R["power_loss"]), "power_loss_L": float(d_L["power_loss"]),
            "power_loss_total": power_loss, "speed": speed, "altitude": altitude, "vz_world": vz_world,
            "z_asym": z_asym, "z_sym": z_sym, "zR": zR, "zL": zL,
            "struct_energy_sum": float(e_R + e_L), "struct_energy_norm": e_sum_norm, "e_sum_max": float(self._e_sum_max),
            **{k: v for k, v in breakdown.items()},
            "coupling_scale": float(self.coupling_scale), "stability_weight": float(self.stability_weight),
            "roll_pitch_limit_deg": float(math.degrees(self.roll_pitch_limit)),
            "twist_enabled": bool(self.twist_enabled), "domain_rand_scale": float(self.randomizer.scale),
            "termination_reason": str(reason) if reason else "",
            "terminal_penalty": terminal_penalty, "survival_ratio": survival_ratio, "terminal_penalty_mult": penalty_mult,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info
