#!/usr/bin/env python3
"""Training-budget sensitivity (revision R1, supplementary; declared 2026-11-08 02:05 before any 2M result existed).

Question: does the ranking of the training paradigms, or the margin over the classical controllers, depend on the
equal 1M-step budget of the pre-specified comparison? Flat SAC and curriculum SAC are retrained from scratch with an
equal budget of 2M steps (the curriculum phases scale with the budget, phases.make_phases), on the same training
seeds as their 1M counterparts (baseline 7, 107; curriculum 17, 117), and evaluated on the same 2 x 50 evaluation
episodes. Two seeds per paradigm is all the remaining compute allows (8-core laptop, ~2.3M steps/h), so the result
is reported as a sensitivity analysis, not as part of the pre-specified comparison, whatever its outcome.

Outputs are isolated in revision_runs/budget2M/ (REVISION_RUNS_DIR), so the primary results cannot be affected.
Scheduling: jobs only use capacity the primary pipeline does not need --
  main training queue has pending jobs -> wait;   main training running -> 2 - (main jobs) slots;
  main evaluation stage running -> 1 slot;          main pipeline ALL DONE -> 2 slots.
Extension jobs run at nice 10, honour revision_runs/PAUSE, and pause below 15% battery.

    python revision/budget_sensitivity.py run       # train (scheduler) + evaluate
    python revision/budget_sensitivity.py compare   # paired 1M vs 2M comparison -> analysis/budget_sensitivity.json
"""

import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)
MAIN = os.path.join(REPO, "revision_runs")
EXT = os.path.join(MAIN, "budget2M")
PY = os.path.join(REPO, ".venv", "bin", "python")
STEPS = 2_000_000
JOBS = [("baseline", 7), ("curriculum", 17), ("baseline", 107), ("curriculum", 117)]
LOG = os.path.join(EXT, "queue.log")


def log(msg):
    os.makedirs(EXT, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def main_state():
    """Return (pending_main, running_main, stage) from the primary pipeline's logs."""
    q = open(os.path.join(MAIN, "queue.log")).read() if os.path.exists(os.path.join(MAIN, "queue.log")) else ""
    a = open(os.path.join(MAIN, "auto_pipeline.log")).read() if os.path.exists(os.path.join(MAIN, "auto_pipeline.log")) else ""
    if "ALL DONE" in a:
        return 0, 0, "done"
    if "[auto] running final_eval.py" in a:
        return 0, 0, "eval"
    rest = q.split("plan=rest")[-1] if "plan=rest" in q else ""
    m = re.search(r"pending=\[(.*?)\]", rest)
    n_total = len(re.findall(r"'([^']+)'", m.group(1))) if m else 13
    started = len(re.findall(r"\] started ", rest)); finished = len(re.findall(r"\] finished ", rest))
    return n_total - started, started - finished, "train"


def battery_low():
    try:
        out = subprocess.check_output(["pmset", "-g", "batt"], text=True)
        pct = int(re.search(r"(\d+)%", out).group(1))
        return "Battery Power" in out and pct < 15
    except Exception:
        return False


def slots():
    pending, running, stage = main_state()
    if stage == "done":
        return 2
    if stage == "eval":
        return 1
    return 0 if pending > 0 else max(0, 2 - running)


def launch(algo, seed):
    os.makedirs(os.path.join(EXT, "console"), exist_ok=True)
    lf = open(os.path.join(EXT, "console", f"{algo}_seed{seed}.log"), "a")
    env = dict(os.environ, REVISION_RUNS_DIR=EXT)
    cmd = ["nice", "-n", "10", PY, "-u", os.path.join(HERE, "train_one.py"), "--algo", algo, "--seed", str(seed),
           "--steps", str(STEPS)]
    return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=REPO)


def done(algo, seed):
    return os.path.exists(os.path.join(EXT, "models", f"{algo}_seed{seed}.zip"))


def run():
    log(f"start: jobs={JOBS} steps={STEPS}")
    todo = [j for j in JOBS if not done(*j)]; procs = {}
    while todo or procs:
        for j, p in list(procs.items()):
            if p.poll() is not None:
                log(f"finished {j} rc={p.returncode}"); procs.pop(j)
        if os.path.exists(os.path.join(MAIN, "PAUSE")) or battery_low():
            open(os.path.join(EXT, "PAUSE"), "w").write("mirrored pause\n")
        elif os.path.exists(os.path.join(EXT, "PAUSE")) and not procs:
            os.remove(os.path.join(EXT, "PAUSE"))     # resume after a mirrored pause (train_one resumes from snapshot)
        while todo and len(procs) < slots() and not os.path.exists(os.path.join(EXT, "PAUSE")):
            j = todo.pop(0); procs[j] = launch(*j); log(f"started {j} (main state {main_state()})")
        time.sleep(60)
    log("training done")
    workers = "4" if main_state()[2] == "done" else "2"
    env = dict(os.environ, REVISION_RUNS_DIR=EXT)
    rc = subprocess.call(["nice", "-n", "5", PY, "-u", os.path.join(HERE, "final_eval.py"), "--workers", workers,
                          "--algos", "baseline", "curriculum"], env=env, cwd=REPO,
                         stdout=open(os.path.join(EXT, "final_eval.log"), "a"), stderr=subprocess.STDOUT)
    log(f"final_eval rc={rc}")
    compare()
    log("ALL DONE")


def compare():
    import numpy as np
    from scipy.stats import wilcoxon
    out = {"doc": __doc__, "steps_ext": STEPS, "pairs": []}
    for algo, seed in JOBS:
        row = {"algo": algo, "seed": seed}
        for cond in ("nominal", "randomized"):
            f1 = os.path.join(MAIN, "eval_final", f"sac_{algo}_seed{seed}_{cond}.json")
            f2 = os.path.join(EXT, "eval_final", f"sac_{algo}_seed{seed}_{cond}.json")
            if not (os.path.exists(f1) and os.path.exists(f2)):
                continue
            e1 = json.load(open(f1))["episodes"]; e2 = json.load(open(f2))["episodes"]
            r1 = np.array([e["rms_yaw_horizon"] for e in e1]); r2 = np.array([e["rms_yaw_horizon"] for e in e2])
            d = r2 - r1
            p = float(wilcoxon(r2, r1).pvalue) if np.any(d != 0) else 1.0
            row[cond] = {"rms_h_1M": float(r1.mean()), "rms_h_2M": float(r2.mean()),
                         "fail_1M": float(np.mean([e["failure"] for e in e1])),
                         "fail_2M": float(np.mean([e["failure"] for e in e2])), "p_wilcoxon": p, "n": int(len(d))}
        out["pairs"].append(row)
    for algo in ("baseline", "curriculum"):
        for cond in ("nominal", "randomized"):
            rows = [r[cond] for r in out["pairs"] if r["algo"] == algo and cond in r]
            if rows:
                out.setdefault("summary", {}).setdefault(algo, {})[cond] = {
                    k: float(np.mean([r[k] for r in rows])) for k in ("rms_h_1M", "rms_h_2M", "fail_1M", "fail_2M")}
    with open(os.path.join(MAIN, "analysis", "budget_sensitivity.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out.get("summary", {}), indent=1))


if __name__ == "__main__":
    {"run": run, "compare": compare}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
