#!/usr/bin/env python3
"""Smoothness fine-tuning of a trained flat-SAC policy (revision R1).

Starts from the final 1M-step flat-SAC policy of one seed (revision_runs/models/baseline_seed<s>.zip, with its
observation normalizer) and continues training on the unchanged flat-SAC task (same environment, randomization,
network and SAC settings) with a smoothness term added:
  * CAPS (Mysore et al. 2021; morphing_glider/training/caps.py): temporal and spatial smoothness of the
    deterministic action in the actor loss (--caps-t, --caps-s, --caps-sigma), and/or
  * a larger weight on the reward's existing actuator-increment penalty -w_jerk ||a_t - a_{t-1}||^2 (--w-jerk;
    flat SAC uses 0.08).
With all smoothness settings at their flat-SAC values the run is a plain continuation (control for extra steps).
Training from scratch with CAPS (revision/caps_study.py) collapsed to the do-nothing policy for one seed; warm
starting avoids the uninformative early critic that makes the smoothness term dominate.
The replay buffer starts empty and is first filled with --prefill transitions of the loaded (stochastic) policy
without gradient updates; learning rate --lr. The total number of environment steps is --steps (prefill included).
Output: <REVISION_RUNS_DIR>/models/smooth_seed<s><tag>.zip (+ .vecnorm.pkl, .meta.json)
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402,F401
from common import RUNS_DIR, N_ENVS, MAX_STEPS_EP  # noqa: E402

MAIN = os.path.join(os.path.dirname(HERE), "revision_runs")


def build_vec(seed, n_envs, w_jerk):
    from stable_baselines3.common.monitor import Monitor
    from morphing_glider.training import infrastructure as inf
    from morphing_glider.environment.wrappers import ProgressiveTwistWrapper
    from morphing_glider.training.task_curriculum import TaskScheduleWrapper, FLAT_SCHEDULE
    phase_dict = {"name": "flat", "twist_factor": 1.0, "rand_scale": 1.0, "ramp_steps": 0, "start_twist_factor": 1.0,
                  "reward_shaper": None}
    ph = FLAT_SCHEDULE[0]

    def thunk(rank):
        def _init():
            env = inf.make_env(seed=int(seed + rank), domain_rand_scale=1.0, max_steps=int(MAX_STEPS_EP), for_eval=False,
                               twist_enabled=True, include_omega_cross=True, roll_pitch_limit_deg=70.0,
                               coupling_scale=1.0, stability_weight=0.03)
            if w_jerk is not None:
                env.unwrapped.reward_computer.w_jerk = float(w_jerk)
            env = Monitor(env)
            env = ProgressiveTwistWrapper(env, phase=phase_dict, twist_factor=1.0, reward_shaper=None, ramp_steps=0)
            env = TaskScheduleWrapper(env)
            env.set_task(ph["targets"], ph["rand"], ph["hold"])
            return env
        return _init

    return inf.make_vec_env([thunk(i) for i in range(int(n_envs))], mode=inf.VECENV_MODE, start_method=inf.SUBPROC_START_METHOD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--prefill", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--caps-t", type=float, default=0.0)
    ap.add_argument("--caps-s", type=float, default=0.0)
    ap.add_argument("--caps-sigma", type=float, default=0.05)
    ap.add_argument("--w-jerk", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--base", default=None, help="policy to start from (default: flat SAC 1M of --seed)")
    a = ap.parse_args()

    from stable_baselines3.common.vec_env import VecNormalize
    from morphing_glider.training.caps import CAPSSAC
    base = a.base or os.path.join(MAIN, "models", f"baseline_seed{a.seed}.zip")
    name = f"smooth_seed{a.seed}{a.tag}"; out = os.path.join(RUNS_DIR, "models"); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    vec = build_vec(a.seed + 9000, N_ENVS, a.w_jerk)
    vn = VecNormalize.load(base.replace(".zip", ".vecnorm.pkl"), vec); vn.training = True; vn.norm_reward = False
    model = CAPSSAC.load(base, env=vn, device="cpu",
                         custom_objects={"learning_rate": float(a.lr), "buffer_size": int(a.steps) + 1000,
                                         "tensorboard_log": None, "learning_starts": 0})
    model.caps_lambda_t, model.caps_lambda_s, model.caps_sigma = float(a.caps_t), float(a.caps_s), float(a.caps_sigma)
    gs = int(model.gradient_steps)
    print(f"[smooth] {name}: base={os.path.basename(base)} steps={a.steps} prefill={a.prefill} lr={a.lr} caps_t={a.caps_t} "
          f"caps_s={a.caps_s} sigma={a.caps_sigma} w_jerk={a.w_jerk} gradient_steps={gs}", flush=True)
    model.gradient_steps = 0
    model.learn(total_timesteps=int(a.prefill), reset_num_timesteps=True, progress_bar=False)
    print(f"[smooth] prefill done ({time.time() - t0:.0f}s), buffer={model.replay_buffer.size()}", flush=True)
    model.gradient_steps = gs
    done = int(a.prefill); chunk = 50_000
    while done < int(a.steps):
        n = min(chunk, int(a.steps) - done)
        model.learn(total_timesteps=n, reset_num_timesteps=False, progress_bar=False)
        done += n
        ep = [e["r"] for e in model.ep_info_buffer] if model.ep_info_buffer else [float("nan")]
        el = [e["l"] for e in model.ep_info_buffer] if model.ep_info_buffer else [float("nan")]
        print(f"[smooth] steps={done}/{a.steps} ep_rew_mean={sum(ep) / len(ep):.1f} ep_len_mean={sum(el) / len(el):.1f} "
              f"wall={time.time() - t0:.0f}s", flush=True)
    model.save(os.path.join(out, name + ".zip")); vn.save(os.path.join(out, name + ".vecnorm.pkl"))
    json.dump({"base": base, "args": vars(a), "wall_h": (time.time() - t0) / 3600, "doc": __doc__},
              open(os.path.join(out, name + ".meta.json"), "w"), indent=1)
    print(f"[done] {name} wall={(time.time() - t0) / 3600:.2f} h", flush=True)
    vn.close()


if __name__ == "__main__":
    main()
