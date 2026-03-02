"""Latent space extraction and MRI visualization for trained SAC policies."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC

from morphing_glider.config import _add_panel_label, _save_fig
from morphing_glider.environment.wrappers import ProgressiveTwistWrapper

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    PCA = None
    TSNE = None


class LatentSpaceExtractor:
    """Extract and store latent activations from SB3 policy network.

    Hooks into the SB3 actor's latent_pi layer (the last hidden layer
    before the final action output) to capture the agent's internal
    representations during evaluation rollouts.

    Args:
        model: Trained SAC model.

    References:
        [GREYDANUS_2018] Visualizing and Understanding Atari Agents.
        [ZAHAVY_2016] Graying the Black Box.
    """

    def __init__(self, model: SAC):
        self.model = model
        self._latents: List[np.ndarray] = []
        self._metadata: List[Dict[str, float]] = []
        self._hook_handle = None
        self._current_latent: Optional[np.ndarray] = None

    def _hook_fn(self, module, input_val, output_val):
        """Forward hook: store the latent activation."""
        self._current_latent = output_val.detach().cpu().numpy().copy()

    def attach(self) -> None:
        """Attach forward hook to actor's latent_pi (last hidden layer)."""
        try:
            actor = self.model.actor
            if hasattr(actor, 'latent_pi') and actor.latent_pi is not None:
                self._hook_handle = actor.latent_pi.register_forward_hook(self._hook_fn)
                print(f"[LatentMRI] Hook attached to actor.latent_pi "
                      f"({type(actor.latent_pi).__name__})")
            elif hasattr(actor, 'features_extractor') and actor.features_extractor is not None:
                self._hook_handle = actor.features_extractor.register_forward_hook(self._hook_fn)
                print(f"[LatentMRI] Hook attached to actor.features_extractor")
            else:
                print("[LatentMRI] WARNING: Could not find hookable layer")
        except Exception as e:
            print(f"[LatentMRI] Hook attach failed: {e!r}")

    def detach(self) -> None:
        """Remove forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def capture(self, flight_condition: Dict[str, float]) -> None:
        """Store the current latent activation with flight metadata.

        Called AFTER model.predict() so that _current_latent holds
        the activations from the most recent forward pass.

        Args:
            flight_condition: Dict with keys like 'yaw_ref', 'roll', 'regime', etc.
        """
        if self._current_latent is not None:
            self._latents.append(self._current_latent.flatten().copy())
            self._metadata.append(dict(flight_condition))

    def get_data(self) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Return all captured latents and metadata.

        Returns:
            Tuple of (latent_array [N, latent_dim], metadata_list).
        """
        if not self._latents:
            return np.array([]).reshape(0, 0), []
        return np.vstack(self._latents), list(self._metadata)

    def reset(self) -> None:
        """Clear captured data."""
        self._latents = []
        self._metadata = []
        self._current_latent = None


class LatentSpaceMRI:
    """Top-Down Macro-Interpretability: Visualize AI cognitive states.

    Projects high-dimensional latent activations into 2D using PCA or t-SNE,
    color-coded by flight regime (stable gliding, gust recovery, turning, etc.).

    This is the 'MRI for AI' -- seeing what the agent's brain is doing
    across different flight conditions, without needing to understand
    individual weights.

    References:
        [ZAHAVY_2016] Graying the Black Box: Understanding DQN.
        [VAN_DER_MAATEN_2008] Visualizing Data using t-SNE.
    """

    @staticmethod
    def classify_flight_regime(info: Dict[str, float]) -> str:
        """Classify current timestep into a flight regime.

        Regimes:
            - 'stable_glide': Low yaw error, low roll/pitch
            - 'turning_stable': High |yaw_ref| with low tracking error
            - 'tracking': Actively reducing yaw error
            - 'gust_recovery': High |roll| or |pitch| perturbation
            - 'stressed': Near attitude limits

        Args:
            info: Step info dict from env.

        Returns:
            String regime label.
        """
        roll = abs(float(info.get("roll", 0.0)))
        pitch = abs(float(info.get("pitch", 0.0)))
        yaw_ref = abs(float(info.get("yaw_ref", 0.0)))
        yaw_err = abs(float(info.get("yaw_error", 0.0)))

        if roll > 0.6 or pitch > 0.6:
            return "stressed"
        if roll > 0.3 or pitch > 0.3:
            return "gust_recovery"
        if yaw_ref > 0.4 and yaw_err < 0.15:
            return "turning_stable"
        if yaw_err > 0.2:
            return "tracking"
        return "stable_glide"

    @staticmethod
    def collect_latents_from_policy(model: SAC, n_episodes: int = 10,
                                     max_steps: int = 200,
                                     obs_rms=None) -> Tuple[np.ndarray, List[Dict]]:
        """Run evaluation episodes and collect latent activations.

        Args:
            model: Trained SAC model.
            n_episodes: Number of episodes to roll out.
            max_steps: Steps per episode.
            obs_rms: VecNormalize obs_rms for observation normalization.

        Returns:
            Tuple of (latent_array [N, latent_dim], metadata_list).
        """
        # Deferred imports to avoid circular dependencies
        from morphing_glider.training.infrastructure import make_env
        from morphing_glider.controllers.sb3_controller import SB3Controller

        extractor = LatentSpaceExtractor(model)
        extractor.attach()

        ctrl = SB3Controller(model, obs_rms=obs_rms)

        for ep in range(n_episodes):
            env = make_env(seed=int(ep + 50000), domain_rand_scale=0.5,
                          max_steps=max_steps, for_eval=True,
                          roll_pitch_limit_deg=65.0, coupling_scale=1.0)
            env = ProgressiveTwistWrapper(env, phase={"name": "mri"},
                                          twist_factor=1.0, reward_shaper=None)
            obs, info = env.reset(seed=int(ep + 50000))
            ctrl.reset()

            for t in range(max_steps):
                action, _ = ctrl.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                regime = LatentSpaceMRI.classify_flight_regime(info)
                flight_cond = {
                    "episode": float(ep), "step": float(t), "regime": regime,
                    "yaw_ref": float(info.get("yaw_ref", 0.0)),
                    "yaw_error": float(info.get("yaw_error", 0.0)),
                    "roll": float(info.get("roll", 0.0)),
                    "pitch": float(info.get("pitch", 0.0)),
                    "speed": float(info.get("speed", 15.0)),
                    "altitude": float(info.get("altitude", 200.0)),
                }
                extractor.capture(flight_cond)

                if terminated or truncated:
                    break

        extractor.detach()
        latents, meta = extractor.get_data()
        print(f"[LatentMRI] Collected {latents.shape[0]} latent vectors "
              f"of dim {latents.shape[1] if latents.ndim > 1 else 0}")
        return latents, meta

    @staticmethod
    def visualize(latents: np.ndarray, metadata: List[Dict],
                  method: str = "pca",
                  save_path: str = "latent_space_mri.png") -> Optional[Dict[str, Any]]:
        """Project and visualize latent space in 2D.

        Args:
            latents: Array of shape (N, latent_dim).
            metadata: List of dicts with 'regime', 'yaw_ref', etc.
            method: 'pca' or 'tsne'.
            save_path: Path to save figure.

        Returns:
            Dict with projection coordinates and explained variance (PCA).
        """
        if latents.size == 0 or len(metadata) == 0:
            print("[LatentMRI] No data to visualize"); return None

        if not _HAS_SKLEARN:
            print("[LatentMRI] sklearn not available; skipping visualization"); return None

        n_samples = latents.shape[0]
        print(f"\n[LatentMRI] Projecting {n_samples} points via {method.upper()}...")

        result_info: Dict[str, Any] = {"method": method, "n_samples": n_samples}

        if method.lower() == "tsne" and n_samples > 50:
            perp = min(30, n_samples // 2)
            projector = TSNE(n_components=2, perplexity=perp, random_state=42,
                            n_iter=1000, init='pca', learning_rate='auto')
            coords = projector.fit_transform(latents)
        else:
            projector = PCA(n_components=min(2, latents.shape[1]), random_state=42)
            coords = projector.fit_transform(latents)
            if hasattr(projector, 'explained_variance_ratio_'):
                evr = projector.explained_variance_ratio_
                result_info["explained_variance_ratio"] = evr.tolist()
                print(f"  PCA explained variance: PC1={evr[0]:.3f}"
                      + (f", PC2={evr[1]:.3f}" if len(evr) > 1 else ""))

        regimes = [m.get("regime", "unknown") for m in metadata]
        unique_regimes = sorted(set(regimes))
        cmap_regimes = {
            "stable_glide": "#2ecc71", "turning_stable": "#3498db",
            "tracking": "#f39c12", "gust_recovery": "#e74c3c",
            "stressed": "#9b59b6", "unknown": "#95a5a6"
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel A: Color by regime
        ax = axes[0]
        for regime in unique_regimes:
            mask = np.array([r == regime for r in regimes])
            color = cmap_regimes.get(regime, "#95a5a6")
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.5,
                      label=f"{regime} ({int(mask.sum())})", color=color)
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_title("Latent Space by Flight Regime")
        ax.legend(fontsize=7, markerscale=2, loc='best')
        _add_panel_label(ax, "A")

        # Panel B: Color by yaw reference
        ax = axes[1]
        yaw_refs = np.array([m.get("yaw_ref", 0.0) for m in metadata])
        sc = ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.5,
                        c=yaw_refs, cmap='RdBu_r', vmin=-0.6, vmax=0.6)
        plt.colorbar(sc, ax=ax, label="Yaw Reference (rad/s)")
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_title("Latent Space by Yaw Target")
        _add_panel_label(ax, "B")

        plt.tight_layout()
        _save_fig(fig, save_path, "AI cognitive state space: flight regime (left) and yaw command (right)")
        plt.show()

        # Print regime distribution
        print(f"\n  Flight regime distribution:")
        for regime in unique_regimes:
            count = sum(1 for r in regimes if r == regime)
            print(f"    {regime}: {count} ({count/len(regimes)*100:.1f}%)")

        return result_info
