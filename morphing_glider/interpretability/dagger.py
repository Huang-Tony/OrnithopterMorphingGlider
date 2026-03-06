"""DAgger imitation learning: train transparent student from opaque expert."""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from morphing_glider.config import DX_RANGE, DY_RANGE, DZ_RANGE, _add_panel_label, _save_fig
from morphing_glider.environment.wrappers import ProgressiveTwistWrapper
from morphing_glider.interpretability.kan import KANPolicyNetwork


class _NormalizedStudentWrapper:
    """Wraps a KAN student to normalize raw observations before predict."""
    def __init__(self, student, obs_rms, clip_obs=10.0):
        self._student = student
        self._obs_rms = obs_rms
        self._clip_obs = float(clip_obs)

    def reset(self):
        self._student.reset()

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        obs = np.asarray(observation, dtype=np.float32)
        if self._obs_rms is not None:
            mean = np.asarray(self._obs_rms.mean, dtype=np.float32)
            var = np.asarray(self._obs_rms.var, dtype=np.float32)
            obs = np.clip((obs - mean) / np.sqrt(var + 1e-8),
                          -self._clip_obs, self._clip_obs).astype(np.float32)
        return self._student.predict(obs, state=state, deterministic=deterministic)


class DAggerDistillation:
    """DAgger imitation learning: train transparent student from opaque expert.

    Iteratively:
      1. Roll out mixed policy (beta * expert + (1-beta) * student)
      2. Query expert for correct actions at ALL student-visited states
      3. Aggregate labelled data with previous dataset
      4. Retrain student on full dataset

    The student can be a KANPolicyNetwork or SymbolicDistiller for
    full interpretability -- resulting in an inspectable controller
    that imitates the black-box SAC agent.

    Args:
        expert: Trained SAC controller (opaque teacher).
        student: KANPolicyNetwork (transparent student).
        n_iterations: Number of DAgger iterations.
        episodes_per_iter: Episodes to collect per iteration.
        max_steps: Steps per episode.
        mix_probability: Initial probability of using expert (beta_0).
        beta_decay: Multiplicative decay for beta each iteration.
        learning_rate: Adam learning rate for student.

    References:
        [ROSS_2011] A Reduction of Imitation Learning to No-Regret
                     Online Learning (DAgger).
        [LASKEY_2017] DART: Noise Injection for Robust Imitation Learning.
    """

    def __init__(self, expert, student: KANPolicyNetwork,
                 n_iterations: int = 10,
                 episodes_per_iter: int = 5,
                 max_steps: int = 200,
                 mix_probability: float = 0.8,
                 beta_decay: float = 0.85,
                 learning_rate: float = 1e-3,
                 obs_rms=None, clip_obs: float = 10.0):
        self.expert = expert
        self.student = student
        self.n_iters = n_iterations
        self.episodes_per_iter = episodes_per_iter
        self.max_steps = max_steps
        self.beta = mix_probability
        self.beta_decay = beta_decay
        self.lr = learning_rate
        self.obs_rms = obs_rms
        self.clip_obs = float(clip_obs)

        self._obs_buffer: List[np.ndarray] = []
        self._act_buffer: List[np.ndarray] = []

        self._device = next(student.parameters()).device

    def _normalize_obs(self, obs):
        if self.obs_rms is None:
            return obs
        mean = np.asarray(self.obs_rms.mean, dtype=np.float32)
        var = np.asarray(self.obs_rms.var, dtype=np.float32)
        return np.clip((obs - mean) / np.sqrt(var + 1e-8),
                       -self.clip_obs, self.clip_obs).astype(np.float32)

    def _collect_iteration(self, iteration: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Collect one DAgger iteration of (observation, expert_action) pairs.

        Rolls out the mixed policy but always queries the expert for labels.
        Observations are normalized using obs_rms (if available) before storage,
        so the student trains in the same observation space as the expert.

        Args:
            iteration: Current DAgger iteration index.

        Returns:
            Tuple of (observations list, expert_actions list).
        """
        # Deferred imports to avoid circular dependencies
        from morphing_glider.training.infrastructure import make_env

        obs_list = []; act_list = []
        beta = self.beta * (self.beta_decay ** iteration)

        for ep in range(self.episodes_per_iter):
            env = make_env(seed=int(iteration * 1000 + ep + 80000),
                          domain_rand_scale=0.5,
                          max_steps=self.max_steps, for_eval=True,
                          roll_pitch_limit_deg=70.0, coupling_scale=1.0)
            env = ProgressiveTwistWrapper(env, phase={"name": "dagger"},
                                          twist_factor=1.0, reward_shaper=None)
            obs, _ = env.reset(seed=int(iteration * 1000 + ep + 80000))
            if hasattr(self.expert, "reset"):
                self.expert.reset()
            self.student.reset()

            for t in range(self.max_steps):
                expert_action, _ = self.expert.predict(obs, deterministic=True)
                # Store normalized obs so student trains in expert's obs space
                obs_norm = self._normalize_obs(obs)
                obs_list.append(obs_norm.copy())
                act_list.append(np.asarray(expert_action, dtype=np.float32).copy())

                if np.random.random() < beta:
                    action = expert_action
                else:
                    # Student predicts from normalized obs
                    action, _ = self.student.predict(obs_norm, deterministic=True)

                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

        return obs_list, act_list

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the full DAgger training loop.

        Args:
            verbose: Print per-iteration stats.

        Returns:
            Dict with training history: loss, n_samples, beta, eval_rms per iter.
        """
        # Deferred import to avoid circular dependencies
        from morphing_glider.training.infrastructure import make_env  # noqa: F401

        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        action_scale_t = torch.tensor(
            [DX_RANGE[1], DY_RANGE[1], DZ_RANGE[1],
             DX_RANGE[1], DY_RANGE[1], DZ_RANGE[1]],
            dtype=torch.float32, device=self._device)

        history: Dict[str, List] = {
            "iteration": [], "loss": [], "n_samples": [],
            "beta": [], "eval_rms": []}

        print(f"\n{'='*80}")
        print("[DAgger] Starting Imitation Learning")
        print(f"  Expert: {type(self.expert).__name__}")
        print(f"  Student: {type(self.student).__name__} "
              f"({sum(p.numel() for p in self.student.parameters())} params)")
        print(f"  Iterations: {self.n_iters}, Episodes/iter: {self.episodes_per_iter}")
        print(f"  Device: {self._device}")
        print(f"{'='*80}")

        for it in range(self.n_iters):
            beta = self.beta * (self.beta_decay ** it)

            new_obs, new_act = self._collect_iteration(it)
            self._obs_buffer.extend(new_obs)
            self._act_buffer.extend(new_act)

            X = torch.tensor(np.array(self._obs_buffer), dtype=torch.float32,
                            device=self._device)
            Y = torch.tensor(np.array(self._act_buffer), dtype=torch.float32,
                            device=self._device)

            Y_scaled = Y / (action_scale_t + 1e-8)
            Y_scaled = torch.clamp(Y_scaled, -1.0, 1.0)

            n_epochs = 20; batch_size = min(256, X.shape[0])
            epoch_losses = []
            self.student.train()

            for epoch in range(n_epochs):
                perm = torch.randperm(X.shape[0], device=self._device)
                total_loss = 0.0; n_batches = 0

                for start in range(0, X.shape[0], batch_size):
                    idx = perm[start:start + batch_size]
                    x_batch = X[idx]
                    y_batch = Y_scaled[idx]

                    pred = self.student(x_batch)
                    loss = loss_fn(pred, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

                epoch_losses.append(total_loss / max(n_batches, 1))

            self.student.eval()

            eval_rms = float("nan")
            try:
                # Deferred import to avoid circular dependencies
                from morphing_glider.evaluation import evaluate_controller
                # Wrap student with obs normalization for proper eval
                eval_student = _NormalizedStudentWrapper(self.student, self.obs_rms, self.clip_obs)
                mets, _ = evaluate_controller(
                    eval_student, n_episodes=3,
                    eval_seed_base=int(90000 + it * 100),
                    domain_rand_scale=0.5, max_steps=self.max_steps,
                    twist_factor=1.0, use_residual_env=False,
                    store_histories=False, roll_pitch_limit_deg=70.0,
                    coupling_scale=1.0, stability_weight=0.03)
                eval_rms = float(np.nanmean(
                    [m.get("rms_yaw_horizon", np.nan) for m in mets]))
            except Exception as e:
                if verbose:
                    print(f"    [DAgger eval failed: {e!r}]")

            history["iteration"].append(it)
            history["loss"].append(float(epoch_losses[-1]))
            history["n_samples"].append(len(self._obs_buffer))
            history["beta"].append(float(beta))
            history["eval_rms"].append(float(eval_rms))

            if verbose:
                print(f"  Iter {it+1}/{self.n_iters}: "
                      f"loss={epoch_losses[-1]:.6f}, "
                      f"samples={len(self._obs_buffer)}, "
                      f"beta={beta:.3f}, "
                      f"eval_rms={eval_rms:.4f}")

        return history

    @staticmethod
    def plot_training_history(history: Dict[str, Any],
                              save_path: str = "dagger_training.png") -> None:
        """Plot DAgger training curves.

        Args:
            history: Output of train().
            save_path: Path to save figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        iters = history["iteration"]

        axes[0].plot(iters, history["loss"], 'o-', color='C0', markersize=4)
        axes[0].set_xlabel("DAgger Iteration")
        axes[0].set_ylabel("MSE Loss")
        axes[0].set_title("Imitation Loss")
        if all(v > 0 for v in history["loss"] if np.isfinite(v)):
            axes[0].set_yscale('log')
        _add_panel_label(axes[0], "A")

        axes[1].plot(iters, history["eval_rms"], 's-', color='C1', markersize=4)
        axes[1].set_xlabel("DAgger Iteration")
        axes[1].set_ylabel("RMS Yaw Error (rad/s)")
        axes[1].set_title("Student Eval Performance")
        _add_panel_label(axes[1], "B")

        axes[2].plot(iters, history["n_samples"], '^-', color='C2', markersize=4)
        ax2r = axes[2].twinx()
        ax2r.plot(iters, history["beta"], 'D--', color='C3', alpha=0.7, markersize=4)
        axes[2].set_xlabel("DAgger Iteration")
        axes[2].set_ylabel("Dataset Size", color='C2')
        ax2r.set_ylabel("Expert Mix (beta)", color='C3')
        axes[2].set_title("Data Aggregation")
        _add_panel_label(axes[2], "C")

        plt.tight_layout()
        _save_fig(fig, save_path,
                  "DAgger distillation: imitation loss (A), student performance (B), data growth (C)")
        plt.show()
