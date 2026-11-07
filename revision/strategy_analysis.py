#!/usr/bin/env python3
"""How do the controllers turn? Steady-turn kinematics and wing shape on the evaluation episodes (revision R1).

For every evaluation episode (nominal condition) the steady-state part of each reference segment with |r_ref| = 0.6
rad/s (from 1.0 s after the switch to the next switch, failures excluded) is collected. Reported per controller,
signed so that positive values point into the commanded turn (sign(r_ref)):
  achieved yaw rate, sideslip beta, bank phi (sign convention: positive = coordinated, i.e. lift tilted toward the
  turn), differential sweep, and the symmetric wing shape (span extension Delta y_s, tip raise Delta z_s).
The roll-due-to-sideslip derivative L_beta is then evaluated (central differences, identify_plant.loads) at the
mean symmetric shape held by each controller, relative to the neutral wing.
Output: revision_runs/analysis/strategy.json
"""

import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from common import RUNS_DIR, dump_json, TRAIN_SEEDS, algo_seed  # noqa: E402
import numpy as np  # noqa: E402

EV = os.path.join(RUNS_DIR, "eval_final")
DT = 0.04; SETTLE = int(1.0 / DT)


def segments(a):
    """Yield (episode, slice, ref) for steady parts of |ref|=0.6 segments of surviving episodes."""
    ref = a["yaw_ref"]
    for i in range(ref.shape[0]):
        r = ref[i]; ok = np.isfinite(r)
        n = int(ok.sum())
        if n < ref.shape[1]:             # failure (episode ended early): excluded
            continue
        start = 0
        for t in range(1, n + 1):
            if t == n or abs(r[t] - r[start]) > 1e-9:
                if abs(abs(r[start]) - 0.6) < 1e-6 and t - start > SETTLE + 5:
                    yield i, slice(start + SETTLE, t), float(np.sign(r[start]))
                start = t


def stats_for(npz_paths):
    acc = {k: [] for k in ("r", "beta_deg", "phi_coord_deg", "ds", "sy", "sz", "dy", "dz")}
    for p in npz_paths:
        a = np.load(p)
        A = a["action"]; obs = a["obs"]
        for i, sl, sg in segments(a):
            act = A[i, sl]; o = obs[i, sl]
            V = np.maximum(o[:, 12], 5.0); beta = np.degrees(np.arcsin(np.clip(o[:, 10] / V, -1, 1)))
            roll = np.degrees(a["roll"][i, sl])
            acc["r"].append(np.mean(a["yaw_rate"][i, sl] * sg))
            acc["beta_deg"].append(np.mean(beta * sg))
            acc["phi_coord_deg"].append(np.mean(-roll * sg))       # positive roll turns toward -y (r<0)
            acc["ds"].append(np.mean(0.5 * (act[:, 0] - act[:, 3]) * sg))
            acc["sy"].append(np.mean(0.5 * (act[:, 1] - act[:, 4])))   # both tips outward
            acc["sz"].append(np.mean(0.5 * (act[:, 2] + act[:, 5])))
            acc["dy"].append(np.mean(0.5 * (act[:, 1] + act[:, 4]) * sg))
            acc["dz"].append(np.mean(0.5 * (act[:, 2] - act[:, 5]) * sg))
    return {k: (float(np.mean(v)) if v else None) for k, v in acc.items()} | {"n_segments": len(acc["r"])}


def l_beta(sy, sz):
    import identify_plant as ip
    a = sy * np.asarray(ip.VIRTUAL["sym_span"], float) + sz * np.asarray(ip.VIRTUAL["sym_dihedral"], float)
    b = math.radians(4.0)
    def M(beta):
        v = ip.V0 * np.array([math.cos(beta), math.sin(beta), 0.0]); return ip.loads(a, v_body=v)[1]
    return float((M(b)[0] - M(-b)[0]) / (2 * b))


def main():
    out = {"doc": __doc__, "controllers": {}}
    groups = {"SAC": [os.path.join(EV, f"sac_baseline_seed{algo_seed('baseline', s)}_nominal.npz") for s in TRAIN_SEEDS],
              "Curriculum": [os.path.join(EV, f"sac_curriculum_seed{algo_seed('curriculum', s)}_nominal.npz") for s in TRAIN_SEEDS],
              "Residual": [os.path.join(EV, f"sac_residual_curriculum_seed{algo_seed('residual_curriculum', s)}_nominal.npz") for s in TRAIN_SEEDS]}
    for f in sorted(glob.glob(os.path.join(EV, "classical_*_nominal.npz"))):
        groups[os.path.basename(f)[10:-12]] = [f]
    Lb0 = l_beta(0.0, 0.0)
    for name, paths in groups.items():
        paths = [p for p in paths if os.path.exists(p)]
        if not paths: continue
        st = stats_for(paths)
        if st["n_segments"]:
            st["L_beta"] = l_beta(st["sy"], st["sz"])
            st["L_beta_ratio"] = st["L_beta"] / Lb0
        st["n_runs"] = len(paths)
        out["controllers"][name] = st
        print(f"{name:12s} n={st['n_segments']:4d} r={st['r']} beta={st['beta_deg']} phi={st['phi_coord_deg']} "
              f"ds={st['ds']} sy={st['sy']} sz={st['sz']} Lb_ratio={st.get('L_beta_ratio')}")
    out["L_beta_neutral"] = Lb0
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "strategy.json"))


if __name__ == "__main__":
    main()
