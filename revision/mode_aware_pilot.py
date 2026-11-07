#!/usr/bin/env python3
"""Pilot: flat SAC trained with the actuator-wing mode randomized in the loop (revision R1, development only).

Motivation: flat SAC departs in every episode when the measured 2.65 Hz wing mode is placed between actuator and
tip (robustness 'struct_mode'); a post-hoc command filter does not help (command_filter.py). Here one flat-SAC seed
(7) is retrained for 1M steps with, in every training episode and with probability 0.5, a second-order mode with
f_n ~ U(2, 4) Hz and zeta ~ U(0.05, 0.15) in the loop (task_curriculum.TaskScheduleWrapper, mode_prob=0.5). It is
compared with the existing flat-SAC seed-7 model on VALIDATION episodes only (seeds 60000+, never the evaluation
set): nominal, randomized, and the measured mode (2.65 Hz, zeta 0.088). Isolated in revision_runs/mode_tests/.

    python revision/mode_aware_pilot.py run | evaluate
"""

import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE); MAIN = os.path.join(REPO, "revision_runs"); EXT = os.path.join(MAIN, "mode_tests")
PY = os.path.join(REPO, ".venv", "bin", "python")
VAL_SEED_BASE, VAL_EPS = 60_000, 50
STRUCT = {"act_wn": 2 * math.pi * 2.65, "act_zeta": 0.088}


def run():
    os.makedirs(os.path.join(EXT, "console"), exist_ok=True)
    env = dict(os.environ, REVISION_RUNS_DIR=EXT)
    with open(os.path.join(EXT, "console", "flat_modeaware_seed7.log"), "a") as lf:
        rc = subprocess.call(["nice", "-n", "10", PY, "-u", os.path.join(HERE, "train_task_curriculum.py"),
                              "--algo", "flat_modeaware", "--seed", "7", "--steps", "1000000"],
                             stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=REPO)
    print("training rc", rc, flush=True)
    evaluate()


def evaluate():
    import numpy as np
    from evaluate import evaluate as ev, load_sac
    models = {"flat_seed7": os.path.join(MAIN, "models", "baseline_seed7.zip"),
              "modeaware_seed7": os.path.join(EXT, "models", "flat_modeaware_seed7.zip")}
    out = {"doc": __doc__, "results": {}}
    for tag, path in models.items():
        if not os.path.exists(path):
            continue
        for cond in ("nominal", "randomized", "struct_mode"):
            kw = dict(STRUCT) if cond == "struct_mode" else {}
            E = ev(load_sac(path), label=tag, condition=cond, rand_scale=1.0 if cond == "randomized" else 0.0,
                   n_episodes=VAL_EPS, seed_base=VAL_SEED_BASE, verbose=True, **kw)["episodes"]
            out["results"][f"{tag}|{cond}"] = {k: float(np.mean([e[k] for e in E])) for k in
                                               ("rms_yaw_horizon", "failure", "mean_tip_speed_mps", "hf_action_power_frac_2Hz")}
    with open(os.path.join(MAIN, "analysis", "mode_aware_pilot.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out["results"].items():
        print(f"{k:32s} RMS@H {v['rms_yaw_horizon']:.3f} fail {100*v['failure']:.0f}% tip {v['mean_tip_speed_mps']:.2f} "
              f"hf {100*v['hf_action_power_frac_2Hz']:.0f}%")


if __name__ == "__main__":
    {"run": run, "evaluate": evaluate}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
