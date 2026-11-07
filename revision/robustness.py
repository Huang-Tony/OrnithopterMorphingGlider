#!/usr/bin/env python3
"""Robustness and envelope experiments (revision R1), nominal physics plus one perturbation at a time.

Model mismatch (beyond the training distribution unless noted):
  tau_100 / tau_140      actuator time constant 0.10 / 0.14 s (hardware slow cluster ~0.10 s)
  struct_mode            servo lag in series with a 2.65 Hz, zeta = 0.088 structural mode (hardware fold test)
  aero_v1                the aerodynamic model of the originally submitted manuscript (soft lift curve, 2-3x drag)
  lift_low               lift slope x0.6 (randomization covers +-25 %)
  mass_inertia           mass x1.4, inertia x1.6 (randomization covers +-25 % / +-35 %)
  gusts_x2               gust intensities x2
  noise_x3               sensor noise x3
  damping_low            lumped rotational damping x0.25 (randomization covers +-50 %)
Reference envelope: the reference alternates between +v and -v, v in {0.15, ..., 0.90} rad/s.
Policies: baseline SAC (all seeds, 20 episodes each) and the two best classical controllers (60 episodes).
Outputs: revision_runs/eval_robust/<tag>.{json,npz}
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from common import RUNS_DIR, TRAIN_SEEDS, QUICK  # noqa: E402

OUT = os.path.join(RUNS_DIR, "eval_robust")
SEED_BASE = 30_000
V1 = {"cd0": 0.03, "k_induced": 0.06, "cl_alpha": 5.0, "cl_max": 1.4, "alpha0": 0.028, "alpha_stall": 0.25,
      "lift_scale": 0.85, "cl_sat_n": None, "post_stall_cl_factor": None}
CONDS = {
    "nominal": {},
    "tau_100": {"actuator_tau": 0.10},
    "tau_140": {"actuator_tau": 0.14},
    "struct_mode": {"act_wn": 2 * math.pi * 2.65, "act_zeta": 0.088},
    "aero_v1": {"phys_override": V1},
    "lift_low": {"phys_override": {"cl_alpha": "x0.6"}},
    "mass_inertia": {"phys_override": {"mass": "x1.4", "Ixx": "x1.6", "Iyy": "x1.6", "Izz": "x1.6"}},
    "gusts_x2": {"phys_override": {"gust_sigma_x": "x2", "gust_sigma_y": "x2", "gust_sigma_z": "x2"}},
    "noise_x3": {"sensor_noise_scale": 3.0},
    "damping_low": {"phys_override": {"d_roll": "x0.25", "d_pitch": "x0.25", "d_yaw": "x0.25"}},
}
ENVELOPE = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]


def _resolve(kw):
    kw = dict(kw)
    po = kw.get("phys_override")
    if po:
        kw["phys_override"] = {k: ((lambda x, f=float(v[1:]): x * f) if isinstance(v, str) and v.startswith("x") else v)
                               for k, v in po.items()}
    return kw


def run_job(kind, name, cond, kw, n_eps, tag):
    n_eps = 1 if QUICK else n_eps
    import torch
    torch.set_num_threads(1)
    from evaluate import evaluate, load_sac
    from classical import make_classical
    if kind == "sac":
        ctrl = load_sac(os.path.join(RUNS_DIR, "models", f"{name}.zip"))
    else:
        tun = json.load(open(os.path.join(RUNS_DIR, "analysis", "classical_tuning.json")))
        ctrl = make_classical(name, **tun[name]["selected"])
    evaluate(ctrl, label=name, condition=cond, rand_scale=0.0, n_episodes=n_eps, seed_base=SEED_BASE,
             out_dir=OUT, tag=tag, verbose=False, **_resolve(kw))
    return tag


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--classical", nargs="+", default=None)
    args = ap.parse_args()
    if args.classical is None:   # two best classical controllers on the nominal final evaluation
        import glob, numpy as np
        rk = []
        for f in glob.glob(os.path.join(RUNS_DIR, "eval_final", "classical_*_nominal.json")):
            d = json.load(open(f)); rk.append((np.mean([e["rms_yaw_horizon"] for e in d["episodes"]]), d["label"]))
        args.classical = [n for _, n in sorted(rk)[:2]] or ["Alloc-ML", "Heuristic"]
    jobs = []
    for cond, kw in CONDS.items():
        for s in TRAIN_SEEDS:
            jobs.append(("sac", f"baseline_seed{s}", cond, kw, 20, f"sac_baseline_seed{s}_{cond}"))
        for c in args.classical:
            jobs.append(("classical", c, cond, kw, 60, f"classical_{c}_{cond}"))
    for v in ENVELOPE:
        kw = {"yaw_targets": [-v, v]}; cond = f"env_{v:.2f}"
        for s in TRAIN_SEEDS:
            jobs.append(("sac", f"baseline_seed{s}", cond, kw, 10, f"sac_baseline_seed{s}_{cond}"))
        for c in args.classical:
            jobs.append(("classical", c, cond, kw, 30, f"classical_{c}_{cond}"))
    jobs = [j for j in jobs if not os.path.exists(os.path.join(OUT, f"{j[-1]}.json"))
            and (j[0] != "sac" or os.path.exists(os.path.join(RUNS_DIR, "models", f"{j[1]}.zip")))]
    print(f"[robustness] {len(jobs)} jobs; classical = {args.classical}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_job, *j): j for j in jobs}
        for f in as_completed(futs):
            try: print("[done]", f.result(), flush=True)
            except Exception as e: print("[FAILED]", futs[f][-1], repr(e), flush=True)


if __name__ == "__main__":
    main()
