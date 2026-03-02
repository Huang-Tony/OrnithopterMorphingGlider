import numpy as np


class ZeroController:
    def reset(self): pass
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        return np.zeros(6, dtype=np.float32), state
