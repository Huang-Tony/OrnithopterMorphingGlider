#!/usr/bin/env python3
"""SAC + CAPS action-smoothness regularization (revision R1): pilot, selection, 5-seed training, evaluation.

Motivation: flat SAC's command is chattery (35% of command power above 2 Hz vs 3% for the heuristic), which is
visible in the videos and destabilizes the loop with the measured wing mode. CAPS (morphing_glider/training/caps.py)
adds temporal and spatial smoothness terms to the actor loss; everything else equals flat SAC.

Selection rule (fixed 2026-11-08 11:45, before any CAPS result): two pilots, lambda_T in {1, 5} (lambda_S =
lambda_T / 2, sigma = 0.05), seed 7, 300k steps, evaluated on VALIDATION episodes (seeds 60000+, 30 nominal); the
smallest lambda_T whose share of command power above 2 Hz is <= 10% is selected (if none, lambda_T = 5). The selected
setting is trained for 5 seeds x 1M steps (the flat-SAC seeds) and evaluated once with the evaluation protocol
(EVAL_SEED_BASE, 50 episodes per seed and condition) and with the measured wing mode (robustness protocol, seed base
30000, 20 episodes per seed). Outputs isolated in revision_runs/caps/; analysis/caps_study.json.
"""

import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE); MAIN = os.path.join(REPO, "revision_runs"); EXT = os.path.join(MAIN, "caps")
PY = os.path.join(REPO, ".venv", "bin", "python")
SEEDS = [7, 107, 207, 307, 407]
PILOT = [1.0, 5.0]
STRUCT = {"act_wn": 2 * math.pi * 2.65, "act_zeta": 0.088}
OUTJ = os.path.join(MAIN, "analysis", "caps_study.json")


def log(msg):
    os.makedirs(EXT, exist_ok=True)
    with open(os.path.join(EXT, "study.log"), "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def train(seed, lam, steps, tag):
    os.makedirs(os.path.join(EXT, "console"), exist_ok=True)
    lf = open(os.path.join(EXT, "console", f"flat_caps_seed{seed}{tag}.log"), "a")
    return subprocess.Popen(["nice", "-n", "5", PY, "-u", os.path.join(HERE, "train_task_curriculum.py"), "--algo", "flat_caps",
                             "--seed", str(seed), "--steps", str(steps), "--tag", tag, "--caps-t", str(lam),
                             "--caps-s", str(lam / 2), "--caps-sigma", "0.05"],
                            stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ, REVISION_RUNS_DIR=EXT), cwd=REPO)


def run_parallel(jobs, width=2):
    todo = list(jobs); procs = []
    while todo or procs:
        procs = [p for p in procs if p.poll() is None]
        while todo and len(procs) < width:
            seed, lam, steps, tag = todo.pop(0); procs.append(train(seed, lam, steps, tag)); log(f"started seed {seed} lam {lam} {tag}")
        time.sleep(30)


def ev(path, cond, seed_base, n):
    import numpy as np
    from evaluate import evaluate, load_sac
    kw = dict(STRUCT) if cond == "struct_mode" else {}
    E = evaluate(load_sac(path), label=os.path.basename(path), condition=cond, rand_scale=1.0 if cond == "randomized" else 0.0,
                 n_episodes=n, seed_base=seed_base, verbose=True, **kw)["episodes"]
    return {k: [float(e[k]) for e in E] for k in ("rms_yaw_horizon", "failure", "hf_action_power_frac_2Hz",
                                                   "mean_tip_speed_mps", "mean_sq_action_increment", "rms_roll_deg")}


def main():
    import numpy as np
    from common import EVAL_SEED_BASE, FINAL_EVAL_EPS
    out = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {"doc": __doc__}
    # 1) pilots
    if "selected_lambda_t" not in out:
        run_parallel([(7, lam, 300_000, f"_pilot{int(lam)}") for lam in PILOT])
        out["pilot"] = {}
        ref = os.path.join(MAIN, "checkpoints", "baseline_seed7", "step_0300000.zip")
        for name, path in [("flat_300k", ref)] + [(f"caps{int(l)}_300k", os.path.join(EXT, "models", f"flat_caps_seed7_pilot{int(l)}.zip")) for l in PILOT]:
            if os.path.exists(path):
                r = ev(path, "nominal", 60_000, 30)
                out["pilot"][name] = {k: float(np.mean(v)) for k, v in r.items()}
                log(f"pilot {name}: {out['pilot'][name]}")
        ok = [l for l in PILOT if out["pilot"].get(f"caps{int(l)}_300k", {}).get("hf_action_power_frac_2Hz", 1.0) <= 0.10]
        out["selected_lambda_t"] = min(ok) if ok else max(PILOT)
        json.dump(out, open(OUTJ, "w"), indent=1); log(f"selected lambda_T = {out['selected_lambda_t']}")
    lam = out["selected_lambda_t"]
    # 2) five seeds x 1M
    run_parallel([(s, lam, 1_000_000, "") for s in SEEDS if not os.path.exists(os.path.join(EXT, "models", f"flat_caps_seed{s}.zip"))])
    log("training done")
    # 3) evaluation protocol + measured wing mode
    out.setdefault("evaluation", {})
    for s in SEEDS:
        path = os.path.join(EXT, "models", f"flat_caps_seed{s}.zip")
        for cond, base, n in (("nominal", EVAL_SEED_BASE, FINAL_EVAL_EPS), ("randomized", EVAL_SEED_BASE, FINAL_EVAL_EPS), ("struct_mode", 30_000, 20)):
            key = f"{s}|{cond}"
            if key not in out["evaluation"] and os.path.exists(path):
                out["evaluation"][key] = ev(path, cond, base, n); json.dump(out, open(OUTJ, "w"), indent=1)
    summ = {}
    for cond in ("nominal", "randomized", "struct_mode"):
        rows = [out["evaluation"][f"{s}|{cond}"] for s in SEEDS if f"{s}|{cond}" in out["evaluation"]]
        if rows:
            summ[cond] = {k: float(np.mean([np.mean(r[k]) for r in rows])) for k in rows[0]}
            summ[cond]["seed_rms_h"] = [float(np.mean(r["rms_yaw_horizon"])) for r in rows]
    out["summary"] = summ; json.dump(out, open(OUTJ, "w"), indent=1)
    log(f"ALL DONE {json.dumps(summ)[:400]}")


if __name__ == "__main__":
    main()
