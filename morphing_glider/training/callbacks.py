"""Training diagnostics callback (revision R1): records SAC internals during training.

Every `every_steps` environment steps it stores the entropy coefficient (alpha), the latest SB3 logger
values (actor/critic/entropy losses, learning rate, episode statistics) and the stochasticity of the
current policy on a probe batch drawn from the replay buffer (per-dimension action std, entropy
estimate). Records accumulate in `self.records` across successive model.learn() calls.
"""

import time
from typing import Any, Dict, List

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class TrainingDiagnosticsCallback(BaseCallback):
    def __init__(self, every_steps: int = 5000, n_probe: int = 512, phase_name: str = ""):
        super().__init__(verbose=0)
        self.every_steps = int(every_steps); self.n_probe = int(n_probe)
        self.phase_name = str(phase_name)
        self.records: List[Dict[str, Any]] = []
        self._last = -10**12; self._t0 = time.time()

    def set_phase(self, name: str) -> None:
        self.phase_name = str(name)

    def _probe_policy(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        m = self.model
        rb = getattr(m, "replay_buffer", None)
        if rb is None or rb.size() <= self.n_probe:
            return out
        try:
            batch = rb.sample(self.n_probe, env=getattr(m, "_vec_normalize_env", None))
            with torch.no_grad():
                mean_actions, log_std, _ = m.actor.get_action_dist_params(batch.observations)
                _, log_prob = m.actor.action_log_prob(batch.observations)
            std = log_std.exp()
            out["policy_std_mean"] = float(std.mean().item())
            out["policy_std_per_dim"] = [float(v) for v in std.mean(0).cpu().numpy()]
            out["policy_log_std_mean"] = float(log_std.mean().item())
            out["policy_entropy_est"] = float(-log_prob.mean().item())
            out["policy_mean_abs_action"] = float(torch.tanh(mean_actions).abs().mean().item())
        except Exception as e:  # diagnostics must never break training
            out["probe_error"] = repr(e)
        return out

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.every_steps:
            return True
        self._last = self.num_timesteps
        m = self.model
        rec: Dict[str, Any] = {"t": int(self.num_timesteps), "wall_s": float(time.time() - self._t0),
                               "phase": self.phase_name}
        log_ent = getattr(m, "log_ent_coef", None)
        if log_ent is not None:
            rec["ent_coef"] = float(log_ent.detach().exp().item())
        elif isinstance(getattr(m, "ent_coef", None), (int, float)):
            rec["ent_coef"] = float(m.ent_coef)
        try:
            for k, v in m.logger.name_to_value.items():
                if np.isscalar(v) and np.isfinite(float(v)):
                    rec[k] = float(v)
        except Exception:
            pass
        rec.update(self._probe_policy())
        self.records.append(rec)
        return True
