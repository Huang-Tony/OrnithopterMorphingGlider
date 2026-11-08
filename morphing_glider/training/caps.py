"""SAC with CAPS action-smoothness regularization (revision R1).

Mysore, Mabsout, Mancuso, Saenko, "Regularizing action policies for smooth control with reinforcement learning",
ICRA 2021. Two terms are added to the actor loss, computed on the deterministic (mean, squashed) action mu(s):
  temporal:  L_T = E || mu(s_t) - mu(s_{t+1}) ||_2          (consecutive states from the replay buffer)
  spatial:   L_S = E || mu(s)   - mu(s + eps) ||_2,  eps ~ N(0, sigma^2 I) in normalized observation units
  actor loss = E[alpha log pi - min Q] + lambda_T L_T + lambda_S L_S.
Everything else (critic, entropy tuning, replay, networks) is identical to stable-baselines3 SAC 2.3.x, whose
train() is reproduced here with only the two terms added.
"""

import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class CAPSSAC(SAC):
    def __init__(self, *args, caps_lambda_t: float = 1.0, caps_lambda_s: float = 0.5, caps_sigma: float = 0.05, **kwargs):
        self.caps_lambda_t = float(caps_lambda_t); self.caps_lambda_s = float(caps_lambda_s)
        self.caps_sigma = float(caps_sigma)
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs, actor_losses, critic_losses, caps_t, caps_s = [], [], [], [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            if self.use_sde:
                self.actor.reset_noise()
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad(); ent_coef_loss.backward(); self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad(); critic_loss.backward(); self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            # ---- CAPS smoothness terms on the deterministic action ----
            mu = self.actor(replay_data.observations, deterministic=True)
            mu_next = self.actor(replay_data.next_observations, deterministic=True)
            noisy = replay_data.observations + self.caps_sigma * th.randn_like(replay_data.observations)
            mu_noisy = self.actor(noisy, deterministic=True)
            l_t = th.norm(mu - mu_next, dim=1).mean()
            l_s = th.norm(mu - mu_noisy, dim=1).mean()
            actor_loss = actor_loss + self.caps_lambda_t * l_t + self.caps_lambda_s * l_s
            caps_t.append(l_t.item()); caps_s.append(l_s.item())
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad(); actor_loss.backward(); self.actor.optimizer.step()
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/caps_temporal", np.mean(caps_t))
        self.logger.record("train/caps_spatial", np.mean(caps_s))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
