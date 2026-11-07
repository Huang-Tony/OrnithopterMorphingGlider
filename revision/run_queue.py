#!/usr/bin/env python3
"""Run a queue of training jobs (one subprocess per algorithm/seed), resumable and pausable.

    python revision/run_queue.py --plan smalltest      # baseline seeds 7 (+checkpoints) and 107
    python revision/run_queue.py --plan rest           # remaining baseline seeds, curriculum, residual
    python revision/run_queue.py --plan all

* Finished runs are skipped; a paused/interrupted run resumes from its snapshot.
* `revision/pause.sh` (creates revision_runs/PAUSE) stops the current run within ~20 s after it saves
  a snapshot; the queue then exits. `revision/resume.sh` removes the flag and relaunches the last plan.
* Battery watchdog: if the Mac runs on battery below --battery-min percent, PAUSE is created automatically.
* Optional --parallel 2 runs two jobs at once. Default device is CPU: on the M3 a CPU job ran 1.6x
  faster than MPS for these small networks, and two concurrent CPU jobs gave ~27% more throughput.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RUNS_DIR, TRAIN_SEEDS, algo_seed, REPO  # noqa: E402

PY = os.path.join(REPO, ".venv", "bin", "python")
PAUSE = os.path.join(RUNS_DIR, "PAUSE")


def plan_jobs(plan):
    small = [("baseline", 7, True), ("baseline", 107, False)]
    rest = ([("baseline", s, False) for s in TRAIN_SEEDS[2:]] +
            [("curriculum", algo_seed("curriculum", s), False) for s in TRAIN_SEEDS] +
            [("residual_curriculum", algo_seed("residual_curriculum", s), False) for s in TRAIN_SEEDS])
    return {"smalltest": small, "rest": rest, "all": small + rest}[plan]


def done(algo, seed):
    n = f"{algo}_seed{seed}"
    return (os.path.exists(os.path.join(RUNS_DIR, "models", f"{n}.zip")) and
            os.path.exists(os.path.join(RUNS_DIR, "logs", f"{n}.train.json")))


def battery_watchdog(threshold, stop_evt):
    while not stop_evt.is_set():
        try:
            out = subprocess.check_output(["pmset", "-g", "batt"], text=True)
            on_batt = "Battery Power" in out
            m = re.search(r"(\d+)%", out); pct = int(m.group(1)) if m else 100
            if on_batt and pct < threshold and not os.path.exists(PAUSE):
                open(PAUSE, "w").write(f"auto-pause: battery {pct}% at {time.ctime()}\n")
                print(f"[watchdog] battery {pct}% on battery power -> PAUSE created", flush=True)
        except Exception:
            pass
        stop_evt.wait(60)


def heartbeat(state):
    try:
        with open(os.path.join(RUNS_DIR, "queue_status.json"), "w") as f:
            json.dump({**state, "updated": time.ctime()}, f, indent=1)
    except Exception:
        pass


def launch(algo, seed, ckpt, device=None):
    os.makedirs(os.path.join(RUNS_DIR, "console"), exist_ok=True)
    log = open(os.path.join(RUNS_DIR, "console", f"{algo}_seed{seed}.log"), "a")
    log.write(f"\n===== launch {time.ctime()} device={device or 'default'} =====\n"); log.flush()
    cmd = [PY, "-u", os.path.join(HERE, "train_one.py"), "--algo", algo, "--seed", str(seed)]
    if ckpt: cmd.append("--checkpoints")
    env = dict(os.environ)
    if device: env["MORPHING_DEVICE"] = device
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="smalltest", choices=["smalltest", "rest", "all"])
    ap.add_argument("--parallel", type=int, default=1, choices=[1, 2])
    ap.add_argument("--battery-min", type=int, default=10)
    ap.add_argument("--device", default="cpu", help="torch device for all jobs (CPU benchmarked fastest)")
    args = ap.parse_args()
    os.makedirs(RUNS_DIR, exist_ok=True)
    json.dump(vars(args), open(os.path.join(RUNS_DIR, "queue_args.json"), "w"))
    stop_evt = threading.Event()
    threading.Thread(target=battery_watchdog, args=(args.battery_min, stop_evt), daemon=True).start()

    pending = [j for j in plan_jobs(args.plan) if not done(j[0], j[1])]
    running = {}   # slot -> (proc, log, job)
    state = {"plan": args.plan, "pending": [f"{a}_seed{s}" for a, s, _ in pending], "running": [], "finished": [],
             "paused": False}
    print(f"[queue] plan={args.plan} pending={state['pending']}", flush=True)
    exit_code = 0
    while pending or running:
        for slot in list(running):
            proc, log, job = running[slot]
            rc = proc.poll()
            if rc is None: continue
            log.close(); del running[slot]
            name = f"{job[0]}_seed{job[1]}"
            if rc == 0: state["finished"].append(name); print(f"[queue] finished {name}", flush=True)
            elif rc == 3: state["paused"] = True; print(f"[queue] paused {name}", flush=True)
            else: exit_code = rc; print(f"[queue] FAILED {name} rc={rc} (see console/{name}.log)", flush=True)
        if os.path.exists(PAUSE) or exit_code:
            if not running: break
        else:
            for slot in range(args.parallel):
                if slot not in running and pending:
                    job = pending.pop(0)
                    device = args.device
                    proc, log = launch(*job, device=device)
                    running[slot] = (proc, log, job)
                    print(f"[queue] started {job[0]}_seed{job[1]} slot={slot} pid={proc.pid}", flush=True)
        state["running"] = [f"{j[0]}_seed{j[1]} (pid {p.pid})" for p, _, j in running.values()]
        state["pending"] = [f"{a}_seed{s}" for a, s, _ in pending]
        heartbeat(state)
        time.sleep(5)
    stop_evt.set(); heartbeat({**state, "running": [], "complete": not pending and not exit_code})
    print(f"[queue] exit (paused={os.path.exists(PAUSE)}, rc={exit_code})", flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
