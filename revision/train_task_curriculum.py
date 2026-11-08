#!/usr/bin/env python3
"""Train one task-distribution-curriculum SAC policy (morphing_glider/training/task_curriculum.py).

Separate from train_one.py so that the running revision pipeline is never affected. Outputs follow the same layout
(models/<name>.zip|.vecnorm.pkl|.meta.json, logs/<name>.train.json, extend/<name>) and use the same seeds as the
corresponding original variants (task_curriculum <- curriculum seeds 17.., residual_task_curriculum <- 27..).

    python revision/train_task_curriculum.py --algo task_curriculum --seed 17 --steps 500000 [--tag _test500k]
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from common import (RUNS_DIR, N_ENVS, MAX_STEPS_EP, EVAL_EVERY, TRAIN_EVAL_EPS, EVAL_SEED_BASE, TOTAL_STEPS,  # noqa: E402
                    dump_json, stamp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["task_curriculum", "residual_task_curriculum", "flat_modeaware",
                                                     "task_curriculum_modeaware", "flat_caps"])
    ap.add_argument("--caps-t", type=float, default=1.0); ap.add_argument("--caps-s", type=float, default=0.5)
    ap.add_argument("--caps-sigma", type=float, default=0.05)
    ap.add_argument("--mode-prob", type=float, default=None, help="probability of a randomized wing mode per episode")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--n-envs", type=int, default=N_ENVS)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--eval-eps", type=int, default=TRAIN_EVAL_EPS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True); os.chdir(RUNS_DIR)
    name = f"{args.algo}_seed{args.seed}{args.tag}"
    model_dir = os.path.join(RUNS_DIR, "models"); log_path = os.path.join(RUNS_DIR, "logs", f"{name}.train.json")
    if os.path.exists(os.path.join(model_dir, f"{name}.zip")) and os.path.exists(log_path):
        print(f"[skip] {name} already trained"); return

    import numpy as np
    from morphing_glider.config import seed_everything
    from morphing_glider.training.task_curriculum import train_task_curriculum_sac, TASK_SCHEDULE, FLAT_SCHEDULE
    from morphing_glider.training.callbacks import TrainingDiagnosticsCallback
    from morphing_glider.training.infrastructure import save_model_and_vecnorm, save_training_checkpoint
    from morphing_glider.training.resume import PauseRequested, clear_snapshot
    from train_one import _keep_final_state
    resume_dir = os.path.join(RUNS_DIR, "resume", name); pause_file = os.path.join(RUNS_DIR, "PAUSE")

    seed_everything(args.seed)
    diag = TrainingDiagnosticsCallback(every_steps=5000)
    t0 = time.time()
    try:
        model, vecnorm, logs = train_task_curriculum_sac(
            total_timesteps=args.steps, seed=args.seed, n_envs=args.n_envs, max_steps=MAX_STEPS_EP,
            eval_every_steps=args.eval_every, eval_episodes=args.eval_eps, eval_seed_base=EVAL_SEED_BASE,
            use_residual=(args.algo == "residual_task_curriculum"), callback=diag,
            schedule=(FLAT_SCHEDULE if args.algo.startswith("flat") else TASK_SCHEDULE),
            mode_prob=(args.mode_prob if args.mode_prob is not None else (0.5 if "modeaware" in args.algo else 0.0)),
            caps=({"lambda_t": args.caps_t, "lambda_s": args.caps_s, "sigma": args.caps_sigma} if "caps" in args.algo else None),
            resume_dir=resume_dir, pause_file=pause_file)
    except PauseRequested as e:
        print(f"[paused] {name}: {e}. Snapshot in {resume_dir}."); sys.exit(3)
    wall = time.time() - t0

    model_path, vecnorm_path = save_model_and_vecnorm(model, vecnorm, out_dir=model_dir, name=name)
    last = logs["evaltrace"][-1] if logs.get("evaltrace") else {}
    meta = {"algo": args.algo, "seed": args.seed, "mean_rmsh": float(last.get("mean_rmsh", np.nan)),
            "use_residual": args.algo == "residual_task_curriculum", "vecnorm_path": vecnorm_path,
            "eval_rand_scale": 1.0, "eval_rpl": 70.0, "total_env_steps": int(model.num_timesteps),
            "wall_clock_s": float(sum(logs.get("wall_clock_s", [])) or wall)}
    save_training_checkpoint(model, model_path, meta)
    dump_json({"name": name, "algo": args.algo, "seed": args.seed, "meta": meta, "stamp": stamp(),
               "actor_params": int(sum(p.numel() for p in model.actor.parameters())),
               "actor_params_without_logstd_head": int(sum(p.numel() for n_, p in model.actor.named_parameters()
                                                           if not n_.startswith("log_std"))),
               "critic_params": int(sum(p.numel() for p in model.critic.parameters())),
               "logs": logs, "diagnostics": diag.records}, log_path)
    print(f"[done] {name}: steps={int(model.num_timesteps)} wall={wall/3600:.2f} h -> {model_path}")
    _keep_final_state(resume_dir, name, clear_snapshot)


if __name__ == "__main__":
    main()
