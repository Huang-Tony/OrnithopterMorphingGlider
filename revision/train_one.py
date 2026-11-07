#!/usr/bin/env python3
"""Train ONE (algorithm, seed) and save model, VecNormalize stats, metadata, training logs and SAC
diagnostics. Skips work that is already complete, so the pipeline is resumable.

Usage:  python revision/train_one.py --algo baseline --seed 7 [--checkpoints] [--steps 1000000]
"""

import argparse
import gc
import multiprocessing
import os
import sys
import time

import common  # noqa: F401  (sets RUN_MODE=paper and sys.path)
from common import (RUNS_DIR, N_ENVS, MAX_STEPS_EP, EVAL_EVERY, TRAIN_EVAL_EPS, EVAL_SEED_BASE,
                    TOTAL_STEPS, dump_json, stamp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["baseline", "curriculum", "residual_curriculum"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--checkpoints", action="store_true", help="save the policy every eval chunk (baseline)")
    ap.add_argument("--n-envs", type=int, default=N_ENVS)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--eval-eps", type=int, default=TRAIN_EVAL_EPS)
    ap.add_argument("--tag", default="", help="optional suffix for smoke tests")
    args = ap.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True); os.chdir(RUNS_DIR)   # TensorBoard dirs are cwd-relative
    name = f"{args.algo}_seed{args.seed}{args.tag}"
    model_dir = os.path.join(RUNS_DIR, "models"); log_path = os.path.join(RUNS_DIR, "logs", f"{name}.train.json")
    if os.path.exists(os.path.join(model_dir, f"{name}.zip")) and os.path.exists(log_path):
        print(f"[skip] {name} already trained"); return

    import numpy as np
    from morphing_glider.config import seed_everything
    from morphing_glider.training.baseline import train_baseline_sac
    from morphing_glider.training.curriculum import train_with_curriculum, summarize_curriculum_progression
    from morphing_glider.training.phases import make_phases
    from morphing_glider.training.callbacks import TrainingDiagnosticsCallback
    from morphing_glider.training.infrastructure import save_model_and_vecnorm, save_training_checkpoint
    from morphing_glider.training.resume import PauseRequested, clear_snapshot
    resume_dir = os.path.join(RUNS_DIR, "resume", name); pause_file = os.path.join(RUNS_DIR, "PAUSE")

    seed_everything(args.seed)
    diag = TrainingDiagnosticsCallback(every_steps=5000)
    t0 = time.time()
    try:
        model, vecnorm, logs = _train(args, name, diag, resume_dir, pause_file)
    except PauseRequested as e:
        print(f"[paused] {name}: {e}. Snapshot in {resume_dir}. Resume with revision/resume.sh")
        sys.exit(3)
    wall = time.time() - t0
    _finish(args, name, model, vecnorm, logs, diag, wall, model_dir, log_path)
    clear_snapshot(resume_dir)


def _train(args, name, diag, resume_dir, pause_file):
    from morphing_glider.training.baseline import train_baseline_sac
    from morphing_glider.training.curriculum import train_with_curriculum, summarize_curriculum_progression
    from morphing_glider.training.phases import make_phases
    if args.algo == "baseline":
        ckpt = os.path.join(RUNS_DIR, "checkpoints", name) if args.checkpoints else None
        model, vecnorm, logs = train_baseline_sac(
            total_timesteps=args.steps, seed=args.seed, n_envs=args.n_envs, max_steps=MAX_STEPS_EP,
            eval_every_steps=args.eval_every, eval_episodes=args.eval_eps, eval_seed_base=EVAL_SEED_BASE,
            eval_domain_rand_scale=1.0, checkpoint_dir=ckpt, callback=diag,
            resume_dir=resume_dir, pause_file=pause_file)
    else:
        model, vecnorm, logs = train_with_curriculum(
            phases=make_phases(args.steps), seed=args.seed, n_envs=args.n_envs, max_steps=MAX_STEPS_EP,
            eval_every_steps=args.eval_every, eval_episodes=args.eval_eps, eval_seed_base=EVAL_SEED_BASE,
            use_residual=(args.algo == "residual_curriculum"),
            total_budget=args.steps, stop_on_gate_fail=False, callback=diag,
            resume_dir=resume_dir, pause_file=pause_file)
        summarize_curriculum_progression(logs)
    return model, vecnorm, logs


def _finish(args, name, model, vecnorm, logs, diag, wall, model_dir, log_path):
    import numpy as np
    from morphing_glider.training.infrastructure import save_model_and_vecnorm, save_training_checkpoint

    model_path, vecnorm_path = save_model_and_vecnorm(model, vecnorm, out_dir=model_dir, name=name)
    last = logs.get("evaltrace", [{}])[-1] if logs.get("evaltrace") else {}
    meta = {"algo": args.algo, "seed": args.seed, "mean_rmsh": float(last.get("mean_rmsh", np.nan)),
            "use_residual": args.algo == "residual_curriculum", "vecnorm_path": vecnorm_path,
            "eval_rand_scale": float(last.get("eval_rand_scale", 1.0)),
            "eval_rpl": float(last.get("eval_rpl", 70.0)), "total_env_steps": int(model.num_timesteps),
            "wall_clock_s": float(sum(logs.get("wall_clock_s", [])) or wall)}
    save_training_checkpoint(model, model_path, meta)
    n_actor = int(sum(p.numel() for p in model.actor.parameters()))
    n_actor_mu = int(sum(p.numel() for n_, p in model.actor.named_parameters() if not n_.startswith("log_std")))
    dump_json({"name": name, "algo": args.algo, "seed": args.seed, "meta": meta, "stamp": stamp(),
               "actor_params": n_actor, "actor_params_without_logstd_head": n_actor_mu,
               "critic_params": int(sum(p.numel() for p in model.critic.parameters())),
               "logs": logs, "diagnostics": diag.records}, log_path)
    print(f"[done] {name}: steps={model.num_timesteps} wall={wall/3600:.2f} h -> {model_path}")
    try: model.env.close()
    except Exception: pass
    del model; gc.collect()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
