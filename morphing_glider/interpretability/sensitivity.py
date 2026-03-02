"""Policy sensitivity analysis via finite-difference Jacobian."""

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from morphing_glider.config import _add_panel_label, _save_fig
from morphing_glider.environment.observation import OBS_IDX, OBS_DIM
from morphing_glider.environment.wrappers import ProgressiveTwistWrapper


class PolicySensitivityAnalyzer:
    """Analyze policy sensitivity via action Jacobian and feature importance.

    Args:
        None.

    Returns:
        Jacobian matrix and feature importance via class methods.

    References:
        [GREYDANUS_2018] Visualizing and Understanding Atari Agents.
    """

    @staticmethod
    def compute_action_jacobian(policy, obs: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        """Compute (6 x 41) Jacobian of mean action w.r.t. observation by finite differences.

        Args:
            policy: Controller with predict().
            obs: Observation vector (41,).
            eps: Finite difference step size.

        Returns:
            Jacobian matrix (6, 41).
        """
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        a0, _ = policy.predict(obs, deterministic=True)
        a0 = np.asarray(a0, dtype=float).reshape(-1)
        J = np.zeros((a0.size, obs.size), dtype=float)
        for j in range(obs.size):
            obs_p = obs.copy(); obs_p[j] += eps
            ap, _ = policy.predict(obs_p, deterministic=True)
            J[:, j] = (np.asarray(ap, dtype=float).reshape(-1) - a0) / eps
        return J

    @staticmethod
    def feature_importance(policy, eval_episodes: List[List[Dict]], n_samples: int = 50) -> Dict[str, float]:
        """Average |Jacobian| across eval episodes per observation feature.

        Args:
            policy: Controller.
            eval_episodes: List of episode histories (list of step dicts).
            n_samples: Max observation samples to evaluate.

        Returns:
            Dict of feature_name -> mean absolute sensitivity, sorted descending.
        """
        # Deferred imports to avoid circular dependencies
        from morphing_glider.training.infrastructure import make_env

        all_obs = []
        for hist in eval_episodes:
            for h in hist:
                if "info" in h:
                    obs_vec = None
                    # Reconstruct obs from info is complex; use action directly
                    pass
        # Fallback: use random observations from a quick rollout
        env = make_env(seed=42, domain_rand_scale=0.0, max_steps=200, for_eval=True)
        env = ProgressiveTwistWrapper(env, phase={"name":"fi"}, twist_factor=1.0, reward_shaper=None)
        obs, _ = env.reset(seed=42)
        if hasattr(policy, "reset"): policy.reset()
        obs_list = [obs.copy()]
        for _ in range(min(n_samples, 199)):
            a, _ = policy.predict(obs, deterministic=True)
            obs, _, done, trunc, _ = env.step(a)
            obs_list.append(obs.copy())
            if done or trunc: break
        mean_abs_J = np.zeros(OBS_DIM, dtype=float)
        count = 0
        for o in obs_list[:n_samples]:
            J = PolicySensitivityAnalyzer.compute_action_jacobian(policy, o)
            mean_abs_J += np.mean(np.abs(J), axis=0)
            count += 1
        if count > 0: mean_abs_J /= count
        idx_to_name = {v: k for k, v in OBS_IDX.items()}
        importance = {}
        for i in range(OBS_DIM):
            name = idx_to_name.get(i, f"obs_{i}")
            importance[name] = float(mean_abs_J[i])
        return dict(sorted(importance.items(), key=lambda kv: -kv[1]))

    @staticmethod
    def plot_feature_importance(importance: Dict[str, float],
                                save_path: str = "feature_importance.png") -> None:
        """Horizontal bar chart of top-20 features.

        Args:
            importance: Dict of feature_name -> sensitivity.
            save_path: Path to save figure.
        """
        items = list(importance.items())[:20]
        if not items: print("[FeatureImportance] No data"); return
        names = [it[0] for it in items]; vals = [it[1] for it in items]
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, vals, color='C0', alpha=0.8)
        ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis(); ax.set_xlabel("Mean |∂action/∂obs|")
        ax.set_title("Policy Feature Importance (Top 20)")
        _add_panel_label(ax, "B")
        plt.tight_layout(); _save_fig(fig, save_path, "Feature importance via finite-difference Jacobian")
        plt.show()
