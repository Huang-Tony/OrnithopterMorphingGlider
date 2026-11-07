#!/usr/bin/env python3
"""Development tests of the task-distribution curriculum (revision R1; not part of the pre-specified comparison).

Runs short task-curriculum trainings in revision_runs/tc_tests/ (isolated) only after the primary 1M training has
finished (one job while the primary evaluation runs, two once it is done), then compares their training-time
evaluation curves with those of the five flat-SAC 1M runs on the same evaluation episodes. Up to any step t, a flat
SAC run does not depend on its total budget (constant learning rate), so the 1M runs are an exact reference.

    python revision/task_curriculum_tests.py run  [--jobs task_curriculum:17:500000 ...]
    python revision/task_curriculum_tests.py compare | validate
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE); MAIN = os.path.join(REPO, "revision_runs"); EXT = os.path.join(MAIN, "tc_tests")
PY = os.path.join(REPO, ".venv", "bin", "python")
DEFAULT_JOBS = ["task_curriculum:17:1000000", "task_curriculum:117:1000000"]
VAL_SEED_BASE = 60_000      # validation episodes: disjoint from tuning (50000+) and evaluation seeds
VAL_EPS = 50


def log(msg):
    os.makedirs(EXT, exist_ok=True)
    with open(os.path.join(EXT, "queue.log"), "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def slots():
    from budget_sensitivity import main_state
    pending, running, stage = main_state()
    return {"done": 2, "eval": 1}.get(stage, 0)       # never during the primary training


def tag_of(steps):
    return f"_t{int(steps) // 1000}k"


def run(jobs):
    jobs = [j.split(":") for j in jobs]; todo = list(jobs); procs = {}
    log(f"start {jobs}")
    while todo or procs:
        for j, p in list(procs.items()):
            if p.poll() is not None:
                log(f"finished {j} rc={p.returncode}"); procs.pop(j)
        while todo and len(procs) < slots() and not os.path.exists(os.path.join(MAIN, "PAUSE")):
            algo, seed, steps = todo.pop(0)
            os.makedirs(os.path.join(EXT, "console"), exist_ok=True)
            lf = open(os.path.join(EXT, "console", f"{algo}_seed{seed}{tag_of(steps)}.log"), "a")
            env = dict(os.environ, REVISION_RUNS_DIR=EXT)
            procs[(algo, seed, steps)] = subprocess.Popen(
                ["nice", "-n", "10", PY, "-u", os.path.join(HERE, "train_task_curriculum.py"), "--algo", algo,
                 "--seed", seed, "--steps", steps, "--tag", tag_of(steps)], stdout=lf, stderr=subprocess.STDOUT,
                env=env, cwd=REPO)
            log(f"started {(algo, seed, steps)}")
        time.sleep(60)
    log("training done"); compare(); validate(); log("ALL DONE")


def curve(path):
    et = json.load(open(path))["logs"]["evaltrace"]
    return {int(e["global_steps"]): float(e["mean_rmsh"]) for e in et if str(e["tag"]).isdigit() or e["tag"] == "init"}


def compare():
    import numpy as np
    flat = [curve(f) for f in sorted(glob.glob(os.path.join(MAIN, "logs", "baseline_seed*.train.json")))]
    old = [curve(f) for f in sorted(glob.glob(os.path.join(MAIN, "logs", "curriculum_seed*.train.json")))]
    out = {"doc": __doc__, "runs": {}}
    for f in sorted(glob.glob(os.path.join(EXT, "logs", "*.train.json"))):
        c = curve(f); name = os.path.basename(f).replace(".train.json", "")
        rows = []
        for t in sorted(c):
            fl = [x[t] for x in flat if t in x]
            rows.append({"step": t, "tc": c[t], "flat_mean": float(np.mean(fl)) if fl else None,
                         "flat_min": float(np.min(fl)) if fl else None, "flat_max": float(np.max(fl)) if fl else None})
        out["runs"][name] = rows
        last = rows[-1]
        print(f"{name:40s} step {last['step']:>7d}: tc {last['tc']:.3f} | flat mean {last['flat_mean']} "
              f"[{last['flat_min']}, {last['flat_max']}]")
        tail = [r for r in rows if r["step"] >= 0.8 * last["step"] and r["flat_mean"] is not None]
        if tail:
            print(f"{'':40s} last 20% of steps: tc {np.mean([r['tc'] for r in tail]):.3f} vs flat "
                  f"{np.mean([r['flat_mean'] for r in tail]):.3f}")
    out["old_curriculum_final"] = [x[max(x)] for x in old if x]
    with open(os.path.join(MAIN, "analysis", "task_curriculum_tests.json"), "w") as f:
        json.dump(out, f, indent=1)


def validate():
    """Decision metric: final models on held-out validation episodes (never the evaluation set)."""
    import numpy as np
    from evaluate import evaluate, load_sac
    models = {f"flat_{os.path.basename(f)[:-4]}": f for f in sorted(glob.glob(os.path.join(MAIN, "models", "baseline_seed*.zip")))}
    models.update({f"oldcur_{os.path.basename(f)[:-4]}": f for f in sorted(glob.glob(os.path.join(MAIN, "models", "curriculum_seed*.zip")))})
    models.update({f"tc_{os.path.basename(f)[:-4]}": f for f in sorted(glob.glob(os.path.join(EXT, "models", "task_curriculum_seed*.zip")))})
    out_p = os.path.join(MAIN, "analysis", "task_curriculum_validation.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for tag, path in models.items():
        for cond, rs in (("nominal", 0.0), ("randomized", 1.0)):
            if f"{tag}|{cond}" in res: continue
            E = evaluate(load_sac(path), label=tag, condition=cond, rand_scale=rs, n_episodes=VAL_EPS,
                         seed_base=VAL_SEED_BASE, verbose=True)["episodes"]
            res[f"{tag}|{cond}"] = {"rms_h": [e["rms_yaw_horizon"] for e in E], "fail": [e["failure"] for e in E]}
            json.dump(res, open(out_p, "w"))
    for grp in ("flat", "oldcur", "tc"):
        for cond in ("nominal", "randomized"):
            v = [np.mean(r["rms_h"]) for k, r in res.items() if k.startswith(grp + "_") and k.endswith(cond)]
            f = [np.mean(r["fail"]) for k, r in res.items() if k.startswith(grp + "_") and k.endswith(cond)]
            if v: print(f"VALIDATION {grp:7s} {cond:10s} RMS@H per seed {np.round(v, 3)} mean {np.mean(v):.3f} | fail {100*np.mean(f):.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "compare", "validate"])
    ap.add_argument("--jobs", nargs="+", default=DEFAULT_JOBS)
    a = ap.parse_args()
    {"run": lambda: run(a.jobs), "compare": compare, "validate": validate}[a.cmd]()
