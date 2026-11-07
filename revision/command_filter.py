#!/usr/bin/env python3
"""Command pre-filter for the learned policy (revision R1, supplementary deployment analysis).

Finding that motivates it: the flat SAC policy departs in every episode when the measured, lightly damped wing mode
(2.65 Hz, zeta 0.088; robustness.py 'struct_mode') is placed between the actuator and the tip, because about 28% of
its command power lies above 2 Hz. A first-order command filter, the standard remedy for flexible structures,
    a_t = alpha * pi(s_t) + (1 - alpha) * a_{t-1},    cutoff f_c = -ln(1 - alpha) / (2 pi dt),
is placed after the trained policy (no retraining). alpha is selected on the classical-controller tuning episodes
(seeds 50000+, 10 per condition) by the mean RMS@H over the nominal, randomized and structural-mode conditions;
the selected filter is then evaluated with the evaluation protocol (final evaluation: 50 episodes per seed and
condition; structural mode: the robustness protocol, 20 episodes per seed). The unfiltered policy (alpha = 1) is
part of the grid. Output: analysis/command_filter.json, revision_runs/eval_filter/.
"""

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from common import RUNS_DIR, EVAL_SEED_BASE, TRAIN_SEEDS, FINAL_EVAL_EPS, dump_json  # noqa: E402
import numpy as np  # noqa: E402

DT = 0.04
ALPHAS = [1.0, 0.6, 0.45, 0.3, 0.2]
STRUCT = {"act_wn": 2 * math.pi * 2.65, "act_zeta": 0.088}
OUT = os.path.join(RUNS_DIR, "eval_filter")


class FilteredController:
    def __init__(self, ctrl, alpha):
        self.ctrl = ctrl; self.alpha = float(alpha); self.prev = None

    def reset(self):
        self.prev = None
        if hasattr(self.ctrl, "reset"): self.ctrl.reset()

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        a, st = self.ctrl.predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)
        a = np.asarray(a, dtype=np.float32)
        self.prev = np.zeros_like(a) if self.prev is None else self.prev
        self.prev = (self.alpha * a + (1.0 - self.alpha) * self.prev).astype(np.float32)
        return self.prev.copy(), st


def cutoff_hz(alpha):
    return float("inf") if alpha >= 1 else -math.log(1 - alpha) / (2 * math.pi * DT)


def _job(args):
    seed, alpha, cond, seed_base, n_eps, tag = args
    import torch; torch.set_num_threads(1)
    from evaluate import evaluate, load_sac
    ctrl = FilteredController(load_sac(os.path.join(RUNS_DIR, "models", f"baseline_seed{seed}.zip")), alpha)
    kw = dict(STRUCT) if cond == "struct_mode" else {}
    rs = 1.0 if cond == "randomized" else 0.0
    res = evaluate(ctrl, label=f"SAC+filter{alpha}", condition=cond, rand_scale=rs, n_episodes=n_eps,
                   seed_base=seed_base, out_dir=(OUT if tag else None), tag=tag, verbose=False, **kw)
    E = res["episodes"]
    keys = ("rms_yaw_horizon", "failure", "mean_tip_speed_mps", "hf_action_power_frac_2Hz", "rms_roll_deg")
    return (seed, alpha, cond), {k: [float(e[k]) for e in E] for k in keys}


def main():
    workers = int(os.environ.get("FILTER_WORKERS", "2"))
    out = {"doc": __doc__, "alphas": ALPHAS, "cutoff_hz": {str(a): cutoff_hz(a) for a in ALPHAS}, "tuning": {}, "evaluation": {}}
    jobs = [(s, a, c, 50_000, 10, None) for s in TRAIN_SEEDS for a in ALPHAS for c in ("nominal", "randomized", "struct_mode")]
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for key, v in ex.map(_job, jobs):
            res[key] = v
    for a in ALPHAS:
        row = {c: {"rms_h": float(np.mean([np.mean(res[(s, a, c)]["rms_yaw_horizon"]) for s in TRAIN_SEEDS])),
                   "failure": float(np.mean([np.mean(res[(s, a, c)]["failure"]) for s in TRAIN_SEEDS]))}
               for c in ("nominal", "randomized", "struct_mode")}
        row["objective"] = float(np.mean([row[c]["rms_h"] for c in ("nominal", "randomized", "struct_mode")]))
        out["tuning"][str(a)] = row
        print(f"alpha={a:<5} fc={cutoff_hz(a):5.2f} Hz  " + "  ".join(f"{c} {row[c]['rms_h']:.3f}/{100*row[c]['failure']:.0f}%"
                                                                for c in ("nominal", "randomized", "struct_mode")), flush=True)
    best = min(ALPHAS, key=lambda a: out["tuning"][str(a)]["objective"])
    out["selected_alpha"] = best; out["selected_cutoff_hz"] = cutoff_hz(best)
    os.makedirs(OUT, exist_ok=True)
    ev = [(s, best, c, EVAL_SEED_BASE, FINAL_EVAL_EPS, f"sac_filtered_seed{s}_{c}") for s in TRAIN_SEEDS for c in ("nominal", "randomized")]
    ev += [(s, best, "struct_mode", 30_000, 20, f"sac_filtered_seed{s}_struct_mode") for s in TRAIN_SEEDS]
    with ProcessPoolExecutor(workers) as ex:
        for (s, a, c), v in ex.map(_job, ev):
            out["evaluation"].setdefault(c, {})[str(s)] = {k: float(np.mean(x)) for k, x in v.items()}
    for c, d in out["evaluation"].items():
        out["evaluation"][c]["mean"] = {k: float(np.mean([d[str(s)][k] for s in TRAIN_SEEDS])) for k in d[str(TRAIN_SEEDS[0])]}
        print(f"EVAL alpha={best} {c}: {out['evaluation'][c]['mean']}", flush=True)
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "command_filter.json"))


if __name__ == "__main__":
    main()
