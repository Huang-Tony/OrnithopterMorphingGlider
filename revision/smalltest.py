#!/usr/bin/env python3
"""Small-test decision (pre-registered before any result was seen, 2026-11-07):

GO      if, in BOTH nominal and randomized conditions, baseline SAC (seeds 7 and 107 pooled, 100 episodes)
        has mean RMS@H <= 0.8 x that of the best tuned classical controller, the 95% BCa CIs do not
        overlap, and the SAC failure rate is not higher than that controller's.
PARTIAL if the criteria hold in only one condition, or the 20% margin holds without CI separation.
NO-GO   otherwise.
SAC evaluations are written to eval_final/ (they are 2 of the 5 final seeds).
"""

import glob
import json
import os

import numpy as np

import common  # noqa: F401
from common import RUNS_DIR, FINAL_EVAL_EPS, EVAL_SEED_BASE, CONDITIONS, dump_json
from evaluate import evaluate, load_sac
from morphing_glider.utils.numeric import bootstrap_mean_ci_bca

EVAL_DIR = os.path.join(RUNS_DIR, "eval_final")


def eval_sac_seed(algo, seed):
    for cond, rs in CONDITIONS.items():
        tag = f"sac_{algo}_seed{seed}_{cond}"
        if os.path.exists(os.path.join(EVAL_DIR, f"{tag}.json")): continue
        ctrl = load_sac(os.path.join(RUNS_DIR, "models", f"{algo}_seed{seed}.zip"))
        kw = {}
        if algo == "residual_curriculum":
            from morphing_glider.training.phases import make_phases
            kw = dict(use_residual=True, residual_limit=make_phases(1)[-1].residual_limit)
        evaluate(ctrl, label=f"{algo}_seed{seed}", condition=cond, rand_scale=rs, n_episodes=FINAL_EVAL_EPS,
                 seed_base=EVAL_SEED_BASE, out_dir=EVAL_DIR, tag=tag, **kw)


def summary(vals):
    v = np.asarray(vals, float); m, lo, hi = bootstrap_mean_ci_bca(v, ci=95.0, seed=123)
    return {"mean": float(m), "lo": float(lo), "hi": float(hi), "n": int(v.size)}


def main():
    for s in (7, 107):
        eval_sac_seed("baseline", s)
    out = {"rule": __doc__, "conditions": {}}
    verdicts = []
    for cond in CONDITIONS:
        sac_eps = []
        for s in (7, 107):
            sac_eps += json.load(open(os.path.join(EVAL_DIR, f"sac_baseline_seed{s}_{cond}.json")))["episodes"]
        sac = {"rms_h": summary([e["rms_yaw_horizon"] for e in sac_eps]),
               "failure": float(np.mean([e["failure"] for e in sac_eps]))}
        classical = {}
        for f in sorted(glob.glob(os.path.join(EVAL_DIR, f"classical_*_{cond}.json"))):
            d = json.load(open(f)); eps = d["episodes"]
            classical[d["label"]] = {"rms_h": summary([e["rms_yaw_horizon"] for e in eps]),
                                     "failure": float(np.mean([e["failure"] for e in eps]))}
        best = min(classical, key=lambda k: classical[k]["rms_h"]["mean"])
        b = classical[best]
        margin_ok = sac["rms_h"]["mean"] <= 0.8 * b["rms_h"]["mean"]
        ci_ok = sac["rms_h"]["hi"] < b["rms_h"]["lo"]
        fail_ok = sac["failure"] <= b["failure"] + 1e-12
        verdict = "GO" if (margin_ok and ci_ok and fail_ok) else ("PARTIAL" if (margin_ok or ci_ok) else "NO-GO")
        verdicts.append(verdict)
        out["conditions"][cond] = {"sac_baseline": sac, "classical": classical, "best_classical": best,
                                   "margin_ok": margin_ok, "ci_ok": ci_ok, "failure_ok": fail_ok, "verdict": verdict,
                                   "rms_ratio_sac_over_best": sac["rms_h"]["mean"] / b["rms_h"]["mean"]}
        print(f"\n== {cond}: SAC RMS@H {sac['rms_h']['mean']:.3f} [{sac['rms_h']['lo']:.3f},{sac['rms_h']['hi']:.3f}] "
              f"fail {100*sac['failure']:.1f}%")
        for k, v in sorted(classical.items(), key=lambda kv: kv[1]["rms_h"]["mean"]):
            print(f"   {k:<10} RMS@H {v['rms_h']['mean']:.3f} [{v['rms_h']['lo']:.3f},{v['rms_h']['hi']:.3f}] fail {100*v['failure']:.1f}%")
        print(f"   best classical = {best}; ratio = {out['conditions'][cond]['rms_ratio_sac_over_best']:.2f}; "
              f"margin={margin_ok} CI={ci_ok} failure={fail_ok} -> {verdict}")
    overall = "GO" if all(v == "GO" for v in verdicts) else ("NO-GO" if all(v == "NO-GO" for v in verdicts) else "PARTIAL")
    out["overall"] = overall
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "smalltest_decision.json"))
    print(f"\nOVERALL: {overall}")


if __name__ == "__main__":
    main()
