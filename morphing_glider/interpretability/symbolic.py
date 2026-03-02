"""Symbolic distillation: polynomial regression from RL policy to interpretable equations."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from morphing_glider.config import DX_RANGE, DY_RANGE, DZ_RANGE
from morphing_glider.environment.observation import OBS_IDX
from morphing_glider.environment.wrappers import ProgressiveTwistWrapper


class SymbolicDistiller:
    """Distill a trained RL policy into symbolic polynomial equations.

    Directly fits polynomial regressors from key observation features
    to each action dimension using least-squares. Reports the discovered
    equations with R-squared values.

    This provides a fully transparent, inspectable control law that
    approximates the black-box RL policy.

    Args:
        polynomial_degree: Degree of polynomial fit per feature.
        key_features: Most important obs features (from Jacobian analysis).

    References:
        [CRANMER_2020] Discovering Symbolic Models from Deep Learning.
        [BRUNTON_2016] Discovering governing equations from data (SINDy).
    """

    def __init__(self, polynomial_degree: int = 3,
                 key_features: Optional[List[str]] = None):
        self.degree = polynomial_degree
        self.key_features = key_features or [
            "omega_r", "yaw_ref", "sin_roll", "cos_roll", "speed"]
        self._coefficients: Optional[np.ndarray] = None
        self._feature_indices: Optional[List[int]] = None
        self._feature_labels: Optional[List[str]] = None

    def collect_expert_data(self, expert, n_episodes: int = 20,
                            max_steps: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Collect (obs, action) pairs from expert policy.

        Args:
            expert: Controller with predict() method.
            n_episodes: Number of rollout episodes.
            max_steps: Steps per episode.

        Returns:
            Tuple of (observations [N, obs_dim], actions [N, 6]).
        """
        # Deferred imports to avoid circular dependencies
        from morphing_glider.training.infrastructure import make_env

        obs_list = []; act_list = []
        for ep in range(n_episodes):
            env = make_env(seed=int(ep + 70000), domain_rand_scale=0.5,
                          max_steps=max_steps, for_eval=True,
                          roll_pitch_limit_deg=65.0, coupling_scale=1.0)
            env = ProgressiveTwistWrapper(env, phase={"name": "distill"},
                                          twist_factor=1.0, reward_shaper=None)
            obs, _ = env.reset(seed=int(ep + 70000))
            if hasattr(expert, "reset"):
                expert.reset()

            for t in range(max_steps):
                action, _ = expert.predict(obs, deterministic=True)
                obs_list.append(obs.copy())
                act_list.append(np.asarray(action, dtype=np.float32).copy())
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

        return np.array(obs_list), np.array(act_list)

    def fit(self, observations: np.ndarray, actions: np.ndarray) -> Dict[str, Any]:
        """Fit polynomial model from observations to actions.

        Builds a polynomial feature matrix from key features (including
        interaction terms), then uses least-squares to fit each action dim.

        Args:
            observations: (N, obs_dim) array.
            actions: (N, action_dim) array.

        Returns:
            Dict with 'equations', 'r_squared' per action, and 'coefficients'.
        """
        self._feature_indices = [OBS_IDX.get(f, 0) for f in self.key_features]
        X = observations[:, self._feature_indices]

        n_feat = X.shape[1]
        poly_features = [np.ones((X.shape[0], 1))]
        feature_labels = ["1"]

        for i in range(n_feat):
            for d in range(1, self.degree + 1):
                poly_features.append(X[:, i:i + 1] ** d)
                name = self.key_features[i]
                feature_labels.append(name if d == 1 else f"{name}^{d}")

        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                poly_features.append((X[:, i] * X[:, j]).reshape(-1, 1))
                feature_labels.append(f"{self.key_features[i]}*{self.key_features[j]}")

        Phi = np.hstack(poly_features)
        self._feature_labels = feature_labels

        action_names = ["p3R_dx", "p3R_dy", "p3R_dz", "p3L_dx", "p3L_dy", "p3L_dz"]
        equations = {}
        r_squared = {}
        self._coefficients = np.zeros((actions.shape[1], Phi.shape[1]))

        print(f"\n{'='*80}")
        print("[SYMBOLIC DISTILLATION] Polynomial Regression")
        print(f"{'='*80}")
        print(f"  Features: {self.key_features}")
        print(f"  Polynomial degree: {self.degree}")
        print(f"  Data points: {X.shape[0]}, Polynomial features: {Phi.shape[1]}")

        for a_idx in range(actions.shape[1]):
            y = actions[:, a_idx]

            try:
                w, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
            except np.linalg.LinAlgError:
                w = np.zeros(Phi.shape[1])

            self._coefficients[a_idx] = w

            y_pred = Phi @ w
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            r_squared[a_idx] = r2

            terms = []
            for fi, (label, coeff) in enumerate(zip(feature_labels, w)):
                if abs(coeff) < 1e-5:
                    continue
                if label == "1":
                    terms.append(f"{coeff:+.5f}")
                else:
                    terms.append(f"{coeff:+.5f}*{label}")

            an = action_names[a_idx] if a_idx < len(action_names) else f"a{a_idx}"
            eq = f"{an} = {' '.join(terms[:10])}"
            if len(terms) > 10:
                eq += f" + ... ({len(terms)} terms)"
            equations[an] = eq

            print(f"\n  {an} (R-squared={r2:.4f}):")
            for term in terms[:6]:
                print(f"    {term}")
            if len(terms) > 6:
                print(f"    ... ({len(terms)} total terms)")

        return {"equations": equations, "r_squared": r_squared,
                "feature_labels": feature_labels, "coefficients": self._coefficients}

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        """Predict using fitted polynomial model.

        Args:
            obs: Observation array (obs_dim,).

        Returns:
            (action, state) tuple.
        """
        if self._coefficients is None or self._feature_indices is None:
            return np.zeros(6, dtype=np.float32), state

        obs = np.asarray(obs, dtype=float).reshape(-1)
        X = obs[self._feature_indices].reshape(1, -1)

        n_feat = X.shape[1]
        poly_features = [np.ones((1, 1))]
        for i in range(n_feat):
            for d in range(1, self.degree + 1):
                poly_features.append(X[:, i:i + 1] ** d)
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                poly_features.append((X[:, i] * X[:, j]).reshape(1, 1))

        Phi = np.hstack(poly_features)
        action = (Phi @ self._coefficients.T).flatten()
        action = np.clip(action,
                        [DX_RANGE[0], DY_RANGE[0], DZ_RANGE[0]] * 2,
                        [DX_RANGE[1], DY_RANGE[1], DZ_RANGE[1]] * 2)
        return action.astype(np.float32), state

    def reset(self):
        pass
