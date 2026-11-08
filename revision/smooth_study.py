#!/usr/bin/env python3
"""Smoothness fine-tuning study (revision R1): pilot, pre-fixed selection, 5 seeds, evaluation.

Why: flat SAC's command is chattery (about 30% of command power above 2 Hz), visible in the videos. A post-hoc
command filter costs accuracy (command_filter.py), and CAPS trained from scratch collapsed to the do-nothing policy
for one seed (caps_study.py, stopped). Here the trained 1M-step flat-SAC policies are fine-tuned for 200k further
steps with a smoothness term (revision/smooth_finetune.py).

Pilot (seed 7, 200k steps each, from baseline_seed7.zip), four runs in parallel:
  caps1        CAPS lambda_T = 1, lambda_S = 0.5, sigma = 0.05
  jerk1        reward actuator-increment weight w_jerk 0.08 -> 1.0
  caps1_jerk1  both
  cont         plain continuation (control for the extra steps; not a candidate)
Selection rule (fixed 2026-11-08 13:10, before any result): validation episodes only (seeds 60000+, 30 nominal and
30 randomized), compared with the 1M base policy on the same episodes. A candidate is eligible if, in both
conditions, its mean RMS@H is <= 1.10 x the base and its failure rate is <= base + 2/30, and its nominal share of
command power above 2 Hz is at most 2/3 of the base's. Among eligible candidates the one with the lowest nominal
share above 2 Hz is selected. If none is eligible, the study stops (no smooth policy is reported).
The selected setting is run for the other four flat-SAC seeds (the seed-7 pilot run is the seed-7 member) and all
five are evaluated once with the evaluation protocol (EVAL_SEED_BASE, 50 episodes per seed and condition) and with
the measured wing mode (seed base 30000, 20 episodes per seed). Outputs in revision_runs/smooth/;
analysis/smooth_study.json.
"""

import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE); MAIN = os.path.join(REPO, "revision_runs"); EXT = os.path.join(MAIN, "smooth")
PY = os.path.join(REPO, ".venv", "bin", "python")
SEEDS = [7, 107, 207, 307, 407]
STEPS = 200_000
VARIANTS = {"caps1": ["--caps-t", "1.0", "--caps-s", "0.5"], "jerk1": ["--w-jerk", "1.0"],
            "caps1_jerk1": ["--caps-t", "1.0", "--caps-s", "0.5", "--w-jerk", "1.0"], "cont": []}
CANDIDATES = ["caps1", "jerk1", "caps1_jerk1"]
STRUCT = {"act_wn": 2 * math.pi * 2.65, "act_zeta": 0.088}
OUTJ = os.path.join(MAIN, "analysis", "smooth_study.json")
KEYS = ("rms_yaw_horizon", "failure", "hf_action_power_frac_2Hz", "mean_tip_speed_mps", "mean_sq_action_increment",
        "rms_roll_deg")


def log(msg):
    os.makedirs(EXT, exist_ok=True)
    with open(os.path.join(EXT, "study.log"), "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def model_path(seed, variant):
    return os.path.join(EXT, "models", f"smooth_seed{seed}_{variant}.zip")


def train(seed, variant):
    os.makedirs(os.path.join(EXT, "console"), exist_ok=True)
    lf = open(os.path.join(EXT, "console", f"smooth_seed{seed}_{variant}.log"), "a")
    return subprocess.Popen(["nice", "-n", "5", PY, "-u", os.path.join(HERE, "smooth_finetune.py"), "--seed", str(seed),
                             "--steps", str(STEPS), "--tag", f"_{variant}"] + VARIANTS[variant],
                            stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ, REVISION_RUNS_DIR=EXT), cwd=REPO)


def run_parallel(jobs, width=4):
    todo = [j for j in jobs if not os.path.exists(model_path(*j))]; procs = []
    while todo or procs:
        procs = [p for p in procs if p.poll() is None]
        while todo and len(procs) < width:
            s, v = todo.pop(0); procs.append(train(s, v)); log(f"started seed {s} {v}")
        time.sleep(20)


def ev(path, cond, seed_base, n):
    from evaluate import evaluate, load_sac
    kw = dict(STRUCT) if cond == "struct_mode" else {}
    E = evaluate(load_sac(path), label=os.path.basename(path), condition=cond, rand_scale=1.0 if cond == "randomized" else 0.0,
                 n_episodes=n, seed_base=seed_base, verbose=True, **kw)["episodes"]
    return {k: [float(e[k]) for e in E] for k in KEYS}


def mean(r):
    import numpy as np
    return {k: float(np.mean(v)) for k, v in r.items()}


def main():
    import numpy as np
    from common import EVAL_SEED_BASE, FINAL_EVAL_EPS
    out = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {"doc": __doc__}
    if "selected" not in out:
        run_parallel([(7, v) for v in VARIANTS])
        log("pilot training done")
        out["pilot"] = {}
        runs = [("base_1M", os.path.join(MAIN, "models", "baseline_seed7.zip"))] + [(v, model_path(7, v)) for v in VARIANTS]
        for name, path in runs:
            if os.path.exists(path):
                out["pilot"][name] = {c: mean(ev(path, c, 60_000, 30)) for c in ("nominal", "randomized")}
                log(f"pilot {name}: {json.dumps(out['pilot'][name])}")
        b = out["pilot"]["base_1M"]; elig = {}
        for v in CANDIDATES:
            p = out["pilot"].get(v)
            if not p: continue
            ok = all(p[c]["rms_yaw_horizon"] <= 1.10 * b[c]["rms_yaw_horizon"] and p[c]["failure"] <= b[c]["failure"] + 2 / 30 + 1e-9
                     for c in ("nominal", "randomized"))
            ok = ok and p["nominal"]["hf_action_power_frac_2Hz"] <= (2 / 3) * b["nominal"]["hf_action_power_frac_2Hz"]
            elig[v] = bool(ok)
        out["eligible"] = elig
        ok = [v for v in CANDIDATES if elig.get(v)]
        out["selected"] = min(ok, key=lambda v: out["pilot"][v]["nominal"]["hf_action_power_frac_2Hz"]) if ok else None
        json.dump(out, open(OUTJ, "w"), indent=1); log(f"eligible {elig}; selected {out['selected']}")
    sel = out["selected"]
    if sel is None:
        log("ALL DONE (no eligible smoothing setting; study stops)"); return
    run_parallel([(s, sel) for s in SEEDS])
    log("training done")
    out.setdefault("evaluation", {})
    for s in SEEDS:
        path = model_path(s, sel)
        for cond, base, n in (("nominal", EVAL_SEED_BASE, FINAL_EVAL_EPS), ("randomized", EVAL_SEED_BASE, FINAL_EVAL_EPS),
                              ("struct_mode", 30_000, 20)):
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
    log(f"ALL DONE {json.dumps(summ)[:500]}")


if __name__ == "__main__":
    main()
