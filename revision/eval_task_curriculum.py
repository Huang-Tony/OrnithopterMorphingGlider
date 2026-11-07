#!/usr/bin/env python3
"""Evaluate the fixed task-curriculum design (5 seeds, 1M steps, revision_runs/tc_tests) with the evaluation protocol
(EVAL_SEED_BASE, 50 episodes per seed and condition), and compare with flat SAC and the original curriculum at the
seed level (same statistics as make_results). Output: revision_runs/tc_tests/eval_final/, analysis/task_curriculum_eval.json."""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: F401
from common import RUNS_DIR, EVAL_SEED_BASE, FINAL_EVAL_EPS, TRAIN_SEEDS, algo_seed
import numpy as np
from scipy import stats

TC = os.path.join(RUNS_DIR, "tc_tests"); OUT = os.path.join(TC, "eval_final")


def main():
    from evaluate import evaluate, load_sac
    from make_results import hier_ci
    os.makedirs(OUT, exist_ok=True)
    res = {"doc": __doc__, "per_seed": {}, "summary": {}, "tests": {}}
    for f in sorted(glob.glob(os.path.join(TC, "models", "task_curriculum_seed*_t1000k.zip"))):
        seed = os.path.basename(f).split("seed")[1].split("_")[0]
        for cond, rs in (("nominal", 0.0), ("randomized", 1.0)):
            tag = f"sac_task_curriculum_seed{seed}_{cond}"
            p = os.path.join(OUT, tag + ".json")
            E = json.load(open(p))["episodes"] if os.path.exists(p) else evaluate(
                load_sac(f), label="task_curriculum", condition=cond, rand_scale=rs, n_episodes=FINAL_EVAL_EPS,
                seed_base=EVAL_SEED_BASE, out_dir=OUT, tag=tag, verbose=True)["episodes"]
            res["per_seed"].setdefault(cond, {})[seed] = {"rms_h": [e["rms_yaw_horizon"] for e in E], "fail": [e["failure"] for e in E]}
    EV = os.path.join(RUNS_DIR, "eval_final")
    for cond in ("nominal", "randomized"):
        tc = [np.array(v["rms_h"]) for v in res["per_seed"].get(cond, {}).values()]
        groups = {"task_curriculum": tc}
        for algo in ("baseline", "curriculum", "residual_curriculum"):
            groups[algo] = [np.array([e["rms_yaw_horizon"] for e in json.load(open(os.path.join(EV, f"sac_{algo}_seed{algo_seed(algo, s)}_{cond}.json")))["episodes"]])
                            for s in TRAIN_SEEDS]
        for g, per in groups.items():
            if per:
                m, lo, hi = hier_ci(per)
                fails = [np.mean(v["fail"]) for v in res["per_seed"][cond].values()] if g == "task_curriculum" else None
                res["summary"][f"{g}|{cond}"] = {"rms_h": [m, lo, hi], "seed_means": [float(x.mean()) for x in per],
                                                 **({"failure": float(np.mean(fails))} if fails else {})}
        a = [x.mean() for x in groups["task_curriculum"]]
        for other in ("baseline", "curriculum"):
            b = [x.mean() for x in groups[other]]
            if len(a) >= 2:
                res["tests"][f"task_curriculum_vs_{other}|{cond}"] = {"p_mannwhitney": float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue),
                                                                     "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b))}
    json.dump(res, open(os.path.join(RUNS_DIR, "analysis", "task_curriculum_eval.json"), "w"), indent=1)
    for k, v in res["summary"].items(): print(k, np.round(v["rms_h"], 3), v.get("failure"))
    print(json.dumps(res["tests"], indent=1))


if __name__ == "__main__":
    main()
