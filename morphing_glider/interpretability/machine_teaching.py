"""Machine Teaching: extract learned control laws and inject into classical controllers."""

from typing import Dict

import numpy as np

from morphing_glider.config import DT, DZ_RANGE, DX_RANGE, DY_RANGE
from morphing_glider.environment.observation import OBS_IDX
from morphing_glider.interpretability.strategy_analyzer import MorphingStrategyAnalyzer


class MachineTeacher:
    """Extracts learned control laws from RL policy and injects into classical controllers.

    Implements Machine Teaching: the RL agent becomes the teacher for
    safety-critical classical controllers by extracting interpretable
    coefficients from its discovered behavior.

    The core idea: if the RL agent learns that
        wing_asymmetry = slope * yaw_target + intercept,
    then that slope IS a new physics coefficient that classical
    controllers can directly use as a feedforward gain.

    References:
        [ZHU_2018] An Overview of Machine Teaching.
        [LENTINK_2007] How swifts control their glide performance.
    """

    @staticmethod
    def extract_learned_coefficient(shapes: Dict[float, Dict]) -> Dict[str, float]:
        """Mathematically extract the linear mapping from yaw_target to wing asymmetry.

        Fits: asymmetry_z = slope * yaw_target + intercept
        via least-squares regression on steady-state morphing data.

        Args:
            shapes: Output of MorphingStrategyAnalyzer.collect_steady_state_shapes.

        Returns:
            Dict with 'slope', 'intercept', 'r_squared', 'residual_std'.
            The slope has units of [m / (rad/s)] -- meters of differential
            wing twist per unit yaw-rate command.
        """
        asym_data = MorphingStrategyAnalyzer.compute_asymmetry_index(shapes)
        if len(asym_data) < 2:
            return {"slope": float("nan"), "intercept": float("nan"),
                    "r_squared": float("nan"), "residual_std": float("nan")}

        targets = np.array([a[0] for a in asym_data])
        asyms = np.array([a[1] for a in asym_data])
        mask = np.isfinite(targets) & np.isfinite(asyms)
        if mask.sum() < 2:
            return {"slope": float("nan"), "intercept": float("nan"),
                    "r_squared": float("nan"), "residual_std": float("nan")}

        t = targets[mask]; a = asyms[mask]
        coeffs = np.polyfit(t, a, 1)
        slope = float(coeffs[0]); intercept = float(coeffs[1])
        predicted = np.polyval(coeffs, t)
        ss_res = float(np.sum((a - predicted) ** 2))
        ss_tot = float(np.sum((a - np.mean(a)) ** 2))
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)
        residual_std = float(np.std(a - predicted))

        print(f"\n{'='*80}")
        print("[MACHINE TEACHING] Extracted Learned Control Law")
        print(f"{'='*80}")
        print(f"  Discovered law: delta_z_asym = {slope:.4f} * yaw_target + {intercept:.4f}")
        print(f"  R-squared = {r_squared:.4f}, residual sigma = {residual_std:.6f}")
        print(f"  Interpretation: For each 1 rad/s yaw rate demand,")
        print(f"    the RL agent applies {abs(slope)*1000:.1f} mm differential wing twist.")
        if r_squared < 0.5:
            print(f"  WARNING: Low R-squared ({r_squared:.3f}). Relationship may be nonlinear.")

        return {"slope": slope, "intercept": intercept,
                "r_squared": r_squared, "residual_std": residual_std}

    @staticmethod
    def inject_into_heuristic(controller, coefficient: Dict[str, float]) -> None:
        """Update VirtualTendonHeuristicController with AI-discovered z_ff_slope.

        Directly sets the feedforward slope used for differential z-twist.

        Args:
            controller: VirtualTendonHeuristicController to update.
            coefficient: Output of extract_learned_coefficient.
        """
        slope = float(coefficient.get("slope", float("nan")))
        if not np.isfinite(slope) or abs(slope) < 1e-6:
            print("[MACHINE TEACHING] Cannot inject: invalid slope"); return

        old_slope = float(controller.z_ff_slope)
        controller.z_ff_slope = float(slope)
        print(f"[MACHINE TEACHING] Heuristic z_ff_slope: {old_slope:.4f} -> {slope:.4f}")

    @staticmethod
    def inject_into_gain_scheduled_pid(controller, coefficient: Dict[str, float]) -> None:
        """Update GainScheduledPIDYawController with AI-derived feedforward scaling.

        Adjusts the overall action_scale based on the AI's discovered
        yaw authority coefficient.

        Args:
            controller: GainScheduledPIDYawController to update.
            coefficient: Output of extract_learned_coefficient.
        """
        slope = float(coefficient.get("slope", float("nan")))
        if not np.isfinite(slope) or abs(slope) < 1e-6:
            print("[MACHINE TEACHING] Cannot inject into GS-PID: invalid slope"); return

        old_scale = float(controller.action_scale)
        new_scale = float(np.clip(abs(slope) * 1.5, 0.05, 0.40))
        controller.action_scale = new_scale
        print(f"[MACHINE TEACHING] GS-PID action_scale: {old_scale:.4f} -> {new_scale:.4f}")

    @staticmethod
    def create_ai_enhanced_pid(coefficient: Dict[str, float],
                                Kp: float = 0.8, Ki: float = 0.05,
                                Kd: float = 0.02, dt: float = DT,
                                action_scale: float = 0.15) -> 'AIEnhancedPIDController':
        """Create a PID controller with AI-learned feedforward term.

        Args:
            coefficient: Output of extract_learned_coefficient.
            Kp, Ki, Kd: PID gains.
            dt: Timestep [s].
            action_scale: Max tip deflection [m].

        Returns:
            AIEnhancedPIDController with integrated feedforward.
        """
        return AIEnhancedPIDController(
            coefficient=coefficient, Kp=Kp, Ki=Ki, Kd=Kd,
            dt=dt, action_scale=action_scale)


class AIEnhancedPIDController:
    """PID yaw controller enhanced with AI-discovered feedforward morphing law.

    Combines classical PID feedback with a learned feedforward mapping:
        action_z = Kff * yaw_ref  (the AI's discovered wing twist law)
        action_x = PID(yaw_error) (classical feedback)

    This hybrid architecture uses the RL agent's discovered physics
    to provide anticipatory control, while PID handles disturbance rejection.

    Args:
        coefficient: Dict from MachineTeacher.extract_learned_coefficient.
        Kp, Ki, Kd: PID gains.
        dt: Timestep [s].
        action_scale: Max control authority [m].

    References:
        [ZHU_2018] An Overview of Machine Teaching.
        [ASTROM_MURRAY_2008] Feedback Systems.
    """
    def __init__(self, coefficient: Dict[str, float],
                 Kp: float = 0.8, Ki: float = 0.05, Kd: float = 0.02,
                 dt: float = DT, action_scale: float = 0.15):
        self.Kff = float(coefficient.get("slope", 0.0))
        self.Kp = float(Kp); self.Ki = float(Ki); self.Kd = float(Kd)
        self.dt = float(dt); self.action_scale = float(action_scale)
        self.integral_limit = 1.0
        self._integral = 0.0; self._prev_error = 0.0
        r2 = float(coefficient.get("r_squared", float("nan")))
        print(f"  [AI-PID] Kff={self.Kff:.4f}, Kp={self.Kp:.4f}, R2={r2:.4f}")

    def reset(self):
        self._integral = 0.0; self._prev_error = 0.0

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        o = np.asarray(obs, dtype=float).reshape(-1)
        r = float(o[OBS_IDX["omega_r"]]); r_ref = float(o[OBS_IDX["yaw_ref"]])
        error = r_ref - r

        self._integral = float(np.clip(self._integral + error * self.dt,
                                        -self.integral_limit, self.integral_limit))
        derivative = (error - self._prev_error) / max(self.dt, 1e-9)
        self._prev_error = float(error)

        # Classical PID feedback -- sweep x-axis
        u_fb = float(np.clip(self.Kp * error + self.Ki * self._integral
                              + self.Kd * derivative, -1.0, 1.0))
        dx = u_fb * self.action_scale

        # AI-learned feedforward -- differential z-axis twist
        dz_R = float(np.clip(self.Kff * r_ref, DZ_RANGE[0], DZ_RANGE[1]))
        dz_L = float(np.clip(-self.Kff * r_ref, DZ_RANGE[0], DZ_RANGE[1]))

        action = np.array([dx, 0.0, dz_R, -dx, 0.0, dz_L], dtype=np.float32)
        return action, state
