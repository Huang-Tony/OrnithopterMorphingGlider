#!/usr/bin/env python3
"""Supplementary classical baseline study (revision R1): allocation about raised-tip operating points.

Motivation. Alloc-ML (classical.py) allocates moments with the control-effectiveness matrix identified at the
neutral wing shape, where the arched spar gives a strong destabilising dihedral effect (lateral_derivatives.py).
Raising the tips reduces that effect five-fold. This study asks whether a classical designer who exploits that
knowledge -- scheduling the allocation about a pre-shaped operating point -- obtains a working multi-loop
controller. It is reported as a supplementary analysis; it does not replace the pre-specified Alloc-ML baseline.

Variant ('Alloc-PS'): identical loops to Alloc-ML (attitude hold, airspeed hold, yaw-rate PI loop, optional
bank-to-turn), but
  * operating point v0 = (sym_span = sy0, sym_dihedral = sz0), applied as a step or ramped over 1 s;
  * B re-identified at v0 by central differences (same procedure as identify_plant.py);
  * optional trim feed-forward: the moment change dM0 = M(v0) - M(0) is subtracted from the demand;
  * yaw demand limited to 80% of the differential-sweep authority at v0 over +-0.3 m.
Grid (72 configurations): sz0 in {0.08, 0.12, 0.15} m, sy0 in {0, 0.1} m, bank_frac in {0 (skid), 1 (bank)},
k_att in {0.5, 1, 2}, trim feed-forward in {off, on}; k_yaw = 1.
Selection: mean RMS@H over the 20 held-out tuning episodes (seeds 50000.., 10 nominal + 10 randomised), the
objective used for every classical baseline (refine_classical.py). The selected configuration is then evaluated
on the 2 x 50 evaluation episodes. Output: revision_runs/analysis/alloc_variants.json and
revision_runs/alloc_variants/*.json|npz.
"""

import functools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from common import RUNS_DIR, EVAL_SEED_BASE, dump_json  # noqa: E402
import numpy as np  # noqa: E402
from classical import AllocationMultiLoop, _att, OBS_IDX, virtual_to_tips, VIRT_NAMES  # noqa: E402

TUNE_SEED_BASE, TUNE_EPS, EVAL_EPS = 50_000, 10, 50
OUT = os.path.join(RUNS_DIR, "alloc_variants")


@functools.lru_cache(maxsize=None)
def _op(sy0, sz0):
    import identify_plant as ip
    a0 = sy0 * np.asarray(ip.VIRTUAL["sym_span"], float) + sz0 * np.asarray(ip.VIRTUAL["sym_dihedral"], float)
    cols = []
    for k in VIRT_NAMES:
        u = np.asarray(ip.VIRTUAL[k], float); h = 0.02
        cols.append((ip.loads(a0 + h * u)[1] - ip.loads(a0 - h * u)[1]) / (2 * h))
    return np.array(cols).T, ip.loads(a0)[1] - ip.loads(np.zeros(6))[1]


class AllocPreshaped(AllocationMultiLoop):
    def __init__(self, sy0=0.0, sz0=0.12, bank_frac=0.0, trim_ff=False, **kw):
        super().__init__(**kw)
        self.v0 = np.array([0, 0, 0, sz0, 0, sy0], float)
        self.B, dM0 = _op(float(sy0), float(sz0))
        self.dM0 = dM0 if trim_ff else np.zeros(3); self.ramp_T = 1.0 if trim_ff else 0.0
        self.lb = np.array([-0.35, -0.15, -0.10, -0.15 - sz0, -0.15, -0.2 - sy0])
        self.ub = np.array([0.35, 0.15, 0.10, 0.15 - sz0, 0.15, 0.2 - sy0])
        self.A_ls = np.vstack([self.W @ self.B, math.sqrt(self.lam) * np.eye(6)])
        self.bank_frac = float(bank_frac)
        self.Mz_max = 0.8 * abs(float(self.B[2, 0])) * 0.3
        self.t = 0.0

    def reset(self):
        super().reset(); self.t = 0.0

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        o, roll, pitch = _att(obs)
        p, q, r = (float(o[OBS_IDX[k]]) for k in ("omega_p", "omega_q", "omega_r"))
        rref = float(o[OBS_IDX["yaw_ref"]]); e = rref - r
        self.i = float(np.clip(self.i + e * self.dt, -1.0, 1.0))
        V = max(float(o[OBS_IDX["speed"]]), 5.0)
        theta_ref = float(np.clip(self.theta_trim + self.k_v * (self.v_ref - V), -0.3, 0.6))
        phi_ref = -math.atan(self.bank_frac * (rref + self.k_i_bank * self.i) * V / self.g)
        phi_ref = float(np.clip(phi_ref, -self.bank_max, self.bank_max))
        step = self.bank_rate * self.dt
        self.phi_cmd = float(np.clip(phi_ref, self.phi_cmd - step, self.phi_cmd + step))
        Mx = -self.I[0] * (self.k_phi * (roll - self.phi_cmd) + self.k_p * p)
        My = -self.I[1] * (self.k_theta * (pitch - theta_ref) + self.k_q * q)
        Mz = float(np.clip(self.I[2] * (self.k_r * e + self.k_i * self.i) + self.Dr * rref, -self.Mz_max, self.Mz_max))
        self.t += self.dt; s = 1.0 if self.ramp_T <= 0 else min(1.0, self.t / self.ramp_T)
        b = np.concatenate([self.W @ (np.array([Mx, My, Mz]) - s * self.dM0), np.zeros(6)])
        lb = self.lb + (1 - s) * self.v0; ub = np.maximum(self.ub + (1 - s) * self.v0, lb + 1e-6)
        v = self._lsq(self.A_ls, b, bounds=(lb, ub), method="bvls").x
        return virtual_to_tips(v + s * self.v0).astype(np.float32), state


def _tune_one(cfg):
    from evaluate import evaluate
    res = {}
    for cond, rs in (("nominal", 0.0), ("randomized", 1.0)):
        E = evaluate(AllocPreshaped(**cfg), label="Alloc-PS", condition=cond, rand_scale=rs, n_episodes=TUNE_EPS,
                     seed_base=TUNE_SEED_BASE, verbose=False)["episodes"]
        res[cond] = {"rms_h": float(np.mean([e["rms_yaw_horizon"] for e in E])),
                     "failure": float(np.mean([e["failure"] for e in E]))}
    res["objective"] = 0.5 * (res["nominal"]["rms_h"] + res["randomized"]["rms_h"])
    return cfg, res


def main():
    from evaluate import evaluate
    os.makedirs(OUT, exist_ok=True)
    grid = [dict(sy0=sy, sz0=sz, bank_frac=bf, k_att=ka, k_yaw=1.0, trim_ff=ff)
            for sz in (0.08, 0.12, 0.15) for sy in (0.0, 0.1) for bf in (0.0, 1.0) for ka in (0.5, 1.0, 2.0)
            for ff in (False, True)]
    rows = []
    with ProcessPoolExecutor(int(os.environ.get("ALLOC_WORKERS", "3"))) as ex:
        for cfg, res in ex.map(_tune_one, grid):
            rows.append({"config": cfg, **res})
            print(f"{json.dumps(cfg):100s} nom {res['nominal']['rms_h']:.3f}/{100*res['nominal']['failure']:3.0f}%  "
                  f"rand {res['randomized']['rms_h']:.3f}/{100*res['randomized']['failure']:3.0f}%", flush=True)
    rows.sort(key=lambda r: r["objective"])
    best = rows[0]["config"]
    out = {"doc": __doc__, "grid": rows, "selected": best, "evaluation": {}}
    for cond, rs in (("nominal", 0.0), ("randomized", 1.0)):
        E = evaluate(AllocPreshaped(**best), label="Alloc-PS", condition=cond, rand_scale=rs, n_episodes=EVAL_EPS,
                     seed_base=EVAL_SEED_BASE, out_dir=OUT, tag=f"classical_Alloc-PS_{cond}", verbose=True)["episodes"]
        out["evaluation"][cond] = {"rms_h_mean": float(np.mean([e["rms_yaw_horizon"] for e in E])),
                                   "failure": float(np.mean([e["failure"] for e in E])), "n": len(E)}
    out["tuning_failure_range"] = [min(min(r["nominal"]["failure"], r["randomized"]["failure"]) for r in rows),
                                   max(max(r["nominal"]["failure"], r["randomized"]["failure"]) for r in rows)]
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "alloc_variants.json"))
    print("[alloc_variants] selected", best, out["evaluation"])


if __name__ == "__main__":
    main()
