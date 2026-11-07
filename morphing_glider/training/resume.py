"""Pause / resume support for long training runs (revision R1).

A run is paused by creating the pause flag file (see revision/pause.sh). The PauseCheckCallback then stops
model.learn() within ~1000 environment steps, the training loop writes a snapshot (model + optimizer
state, replay buffer, VecNormalize statistics and all loop variables) and raises PauseRequested. Re-running
the same command resumes from the snapshot. Snapshots are also written after every evaluation chunk, so
an unplanned shutdown loses at most one chunk. Resumed runs are not bit-identical to uninterrupted ones
(environments are reset and RNG streams differ), which is noted in the run log.
"""

import os
import shutil
from typing import Any, Dict, Optional

import torch
from stable_baselines3.common.callbacks import BaseCallback


class PauseRequested(Exception):
    """Raised after a snapshot has been written because the pause flag was set."""


def pause_flag_set(pause_file: Optional[str]) -> bool:
    return bool(pause_file) and os.path.exists(pause_file)


class PauseCheckCallback(BaseCallback):
    def __init__(self, pause_file: Optional[str], every_steps: int = 1000):
        super().__init__(verbose=0)
        self.pause_file = pause_file; self.every_steps = int(every_steps)
        self._last = -10**12; self.paused = False

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.every_steps:
            return True
        self._last = self.num_timesteps
        if pause_flag_set(self.pause_file):
            self.paused = True
            return False
        return True


def _cpu(obj):
    """Recursively move tensors to CPU so snapshots load on any device."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu(v) for v in obj)
    return obj


def save_snapshot(dirpath: str, *, model, state: Dict[str, Any]) -> None:
    """Atomically write model (+optimizers), replay buffer and loop state to dirpath."""
    tmp = dirpath + ".tmp"; old = dirpath + ".old"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp, exist_ok=True)
    model.save(os.path.join(tmp, "model.zip"))
    model.save_replay_buffer(os.path.join(tmp, "replay.pkl"))
    torch.save(_cpu(state), os.path.join(tmp, "state.pt"))
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(dirpath):
        os.rename(dirpath, old)
    os.rename(tmp, dirpath)
    shutil.rmtree(old, ignore_errors=True)


def load_snapshot(dirpath: Optional[str]) -> Optional[Dict[str, Any]]:
    if not dirpath:
        return None
    for cand in (dirpath, dirpath + ".old"):   # .old survives a crash during the atomic swap
        sp = os.path.join(cand, "state.pt")
        if os.path.exists(sp) and os.path.exists(os.path.join(cand, "model.zip")):
            return {"dir": cand, "model": os.path.join(cand, "model.zip"),
                    "replay": os.path.join(cand, "replay.pkl"),
                    "state": torch.load(sp, map_location="cpu", weights_only=False)}
    return None


def clear_snapshot(dirpath: Optional[str]) -> None:
    if dirpath:
        for cand in (dirpath, dirpath + ".old", dirpath + ".tmp"):
            shutil.rmtree(cand, ignore_errors=True)
