"""Morphing strategy analysis: wing asymmetry vs yaw target."""

from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from morphing_glider.config import _add_panel_label, _save_fig
from morphing_glider.environment.wrappers import ProgressiveTwistWrapper


class MorphingStrategyAnalyzer:
    """Analyzes learned morphing strategies for interpretability.

    Args:
        None.

    Returns:
        Wing shape data and asymmetry indices via class methods.

    References:
        [LENTINK_2007] How swifts control their glide performance.
    """

    @staticmethod
    def collect_steady_state_shapes(policy, env_factory: Callable,
                                     yaw_targets: Sequence[float] = (-0.6, -0.3, 0.0, 0.3, 0.6),
                                     n_episodes: int = 5) -> Dict[float, Dict[str, Any]]:
        """For each yaw target, collect wing tip positions during last 30 steps.

        Args:
            policy: Controller.
            env_factory: Not used (creates envs directly).
            yaw_targets: Targets to evaluate.
            n_episodes: Episodes per target.

        Returns:
            Dict[target] -> {p3_R: array, p3_L: array, p3_R_std, p3_L_std}.
        """
        # Deferred imports to avoid circular dependencies
        from morphing_glider.training.infrastructure import make_env

        def run_episode(env, controller, *, deterministic=True, seed=None):
            obs, info = env.reset(seed=seed)
            if hasattr(controller, "reset") and callable(getattr(controller, "reset")):
                controller.reset()
            T = int(env.unwrapped.max_steps)
            hist = []
            for t in range(T):
                action, _ = controller.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                hist.append({"t": t, "info": dict(info),
                             "terminated": bool(terminated), "truncated": bool(truncated)})
                if terminated or truncated:
                    break
            return hist

        shapes = {}
        for tgt in yaw_targets:
            p3_R_all = []; p3_L_all = []
            for ep in range(n_episodes):
                env = make_env(seed=int(ep+1000*hash(str(tgt))%9999), domain_rand_scale=0.0,
                               max_steps=200, for_eval=True, roll_pitch_limit_deg=65.0)
                env.unwrapped.yaw_targets = [float(tgt)]
                env = ProgressiveTwistWrapper(env, phase={"name":"shape"}, twist_factor=1.0, reward_shaper=None)
                hist = run_episode(env, policy, deterministic=True, seed=int(ep))
                if len(hist) > 30:
                    for h in hist[-30:]:
                        info = h.get("info", {})
                        p3_R_all.append([float(info.get("zR", 0.0))])
                        p3_L_all.append([float(info.get("zL", 0.0))])
            if p3_R_all:
                p3_R_arr = np.array(p3_R_all); p3_L_arr = np.array(p3_L_all)
                shapes[float(tgt)] = {
                    "p3_R_z_mean": float(np.mean(p3_R_arr)), "p3_R_z_std": float(np.std(p3_R_arr)),
                    "p3_L_z_mean": float(np.mean(p3_L_arr)), "p3_L_z_std": float(np.std(p3_L_arr)),
                }
            else:
                shapes[float(tgt)] = {"p3_R_z_mean": float("nan"), "p3_R_z_std": float("nan"),
                                       "p3_L_z_mean": float("nan"), "p3_L_z_std": float("nan")}
        return shapes

    @staticmethod
    def compute_asymmetry_index(shapes: Dict[float, Dict]) -> List[Tuple[float, float]]:
        """Compute asym_z = mean(p3_R_z) - mean(p3_L_z) per target.

        Args:
            shapes: Output of collect_steady_state_shapes.

        Returns:
            List of (yaw_target, asym_z) tuples.
        """
        result = []
        for tgt in sorted(shapes.keys()):
            s = shapes[tgt]
            asym = float(s["p3_R_z_mean"]) - float(s["p3_L_z_mean"])
            result.append((float(tgt), float(asym)))
        return result

    @staticmethod
    def plot_asymmetry_curve(shapes: Dict, save_path: str = "asymmetry_curve.png") -> None:
        """Plot asym_z vs yaw_target with linear fit.

        Args:
            shapes: Output of collect_steady_state_shapes.
            save_path: Path to save figure.
        """
        asym_data = MorphingStrategyAnalyzer.compute_asymmetry_index(shapes)
        if not asym_data: print("[Asymmetry] No data"); return
        targets = np.array([a[0] for a in asym_data])
        asyms = np.array([a[1] for a in asym_data])
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.scatter(targets, asyms, s=40, zorder=5, color='C0')
        # Error bars from std
        errs = []
        for tgt in targets:
            s = shapes.get(float(tgt), {})
            errs.append(float(s.get("p3_R_z_std", 0.0)) + float(s.get("p3_L_z_std", 0.0)))
        ax.errorbar(targets, asyms, yerr=errs, fmt='none', ecolor='gray', alpha=0.5)
        # Linear fit
        mask = np.isfinite(targets) & np.isfinite(asyms)
        if mask.sum() >= 2:
            coeffs = np.polyfit(targets[mask], asyms[mask], 1)
            x_fit = np.linspace(targets.min(), targets.max(), 50)
            ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color='C1', label=f'fit: slope={coeffs[0]:.3f}')
            ax.legend()
        ax.set_xlabel("Yaw Target (rad/s)"); ax.set_ylabel("Wing Asymmetry $\\Delta z_R - \\Delta z_L$ (m)")
        ax.set_title("Learned Morphing: Wing Asymmetry vs Yaw Target")
        _add_panel_label(ax, "A"); ax.grid(True, alpha=0.2)
        plt.tight_layout(); _save_fig(fig, save_path, "Asymmetry index shows learned differential morphing strategy")
        plt.show()
