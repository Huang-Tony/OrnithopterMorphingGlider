#!/usr/bin/env python3
"""Final evaluation of every trained policy (algorithms x seeds x {nominal, randomized}), 50 episodes each,
with full per-step data capture (revision_runs/eval_final/sac_<algo>_seed<s>_<cond>.{json,npz}).
Runs jobs in parallel worker processes; finished outputs are skipped (resumable).

    python revision/final_eval.py [--workers 4] [--algos baseline curriculum residual_curriculum]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from common import RUNS_DIR, TRAIN_SEEDS, FINAL_EVAL_EPS, EVAL_SEED_BASE, CONDITIONS, algo_seed  # noqa: E402

EVAL_DIR = os.path.join(RUNS_DIR, "eval_final")


def run_job(algo, seed, cond):
    import torch
    torch.set_num_threads(1)
    from evaluate import evaluate, load_sac
    tag = f"sac_{algo}_seed{seed}_{cond}"
    ctrl = load_sac(os.path.join(RUNS_DIR, "models", f"{algo}_seed{seed}.zip"))
    kw = {}
    if algo == "residual_curriculum":
        from morphing_glider.training.phases import make_phases
        kw = dict(use_residual=True, residual_limit=make_phases(1)[-1].residual_limit)
    evaluate(ctrl, label=f"{algo}_seed{seed}", condition=cond, rand_scale=CONDITIONS[cond], n_episodes=FINAL_EVAL_EPS,
             seed_base=EVAL_SEED_BASE, out_dir=EVAL_DIR, tag=tag, verbose=False, **kw)
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--algos", nargs="+", default=["baseline", "curriculum", "residual_curriculum"])
    args = ap.parse_args()
    jobs = []
    for algo in args.algos:
        for s in TRAIN_SEEDS:
            seed = algo_seed(algo, s)
            if not os.path.exists(os.path.join(RUNS_DIR, "models", f"{algo}_seed{seed}.zip")):
                print(f"[skip] model missing: {algo}_seed{seed}"); continue
            for cond in CONDITIONS:
                if not os.path.exists(os.path.join(EVAL_DIR, f"sac_{algo}_seed{seed}_{cond}.json")):
                    jobs.append((algo, seed, cond))
    print(f"[final_eval] {len(jobs)} jobs, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_job, *j): j for j in jobs}
        for f in as_completed(futs):
            try: print("[done]", f.result(), flush=True)
            except Exception as e: print("[FAILED]", futs[f], repr(e), flush=True)


if __name__ == "__main__":
    main()
