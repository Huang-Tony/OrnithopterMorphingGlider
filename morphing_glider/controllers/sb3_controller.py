import numpy as np


class SB3Controller:
    """Wraps a trained SB3 SAC model for evaluation with VecNormalize obs stats."""
    def __init__(self, model, *, obs_rms=None, clip_obs=10.0):
        self.model = model; self.obs_rms = obs_rms; self.clip_obs = float(clip_obs); self.eps = 1e-8

    def reset(self): pass

    def _normalize_obs(self, obs):
        if self.obs_rms is None: return obs
        mean = np.asarray(self.obs_rms.mean); var = np.asarray(self.obs_rms.var)
        return np.clip((obs - mean) / np.sqrt(var + self.eps), -self.clip_obs, self.clip_obs).astype(np.float32)

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        obs_n = self._normalize_obs(np.asarray(observation, dtype=np.float32))
        action, _ = self.model.predict(obs_n, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32), state
