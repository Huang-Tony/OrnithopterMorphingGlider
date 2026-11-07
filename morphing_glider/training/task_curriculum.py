"""Task-distribution curriculum for SAC (revision R1).

Design principle. Off-policy SAC learns from a replay buffer that mixes transitions collected over many phases.
A curriculum that changes the *dynamics* (e.g. scaled aerodynamic moments), the *reward*, the *termination
rule* or the *action mapping* between phases therefore trains the critic on transitions that are not valid
samples of the target MDP; the bias persists for as long as these transitions stay in the buffer. Here every phase
is the target MDP itself, restricted to an easier distribution over its exogenous variables only:
  * the set of commanded yaw rates (the reference is part of the observation, so a transition collected under a
    smaller reference is a valid transition of the target MDP),
  * the range of the randomised physical parameters, wind and gusts (a narrower range is a subset of the target
    hidden-parameter distribution),
  * the hold time between reference switches (fewer switches early on).
Dynamics, reward, termination, action mapping, network, optimiser and learning rate are identical to flat SAC in
every phase, the replay buffer is never reset, no checkpoint is selected, and the last phase is exactly the flat-SAC
training distribution. Phases are allocated fixed fractions of the step budget, so the method has no gates that
would need held-out episodes. (Curriculum as a sequence of task distributions: Bengio et al. 2009; Narvekar et al.
2020; goal/start-distribution curricula: Florensa et al. 2017.)
"""

import copy
import gc
import time
from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
import numpy as np

from morphing_glider.config import DEFAULT_YAW_TARGETS, HOLD_RANGE_STEPS

TASK_SCHEDULE = [
    dict(name="tc1_small_refs", frac=0.10, targets=(-0.15, 0.0, 0.15), rand=0.0, hold=(100, 150)),
    dict(name="tc2_moderate_refs", frac=0.15, targets=(-0.3, -0.15, 0.0, 0.15, 0.3), rand=0.33, hold=(80, 120)),
    dict(name="tc3_large_refs", frac=0.20, targets=(-0.45, -0.3, 0.0, 0.3, 0.45), rand=0.67, hold=tuple(HOLD_RANGE_STEPS)),
    dict(name="tc4_target_task", frac=0.55, targets=tuple(DEFAULT_YAW_TARGETS), rand=1.0, hold=tuple(HOLD_RANGE_STEPS)),
]
RESIDUAL_LIMIT = np.array([0.25, 0.12, 0.15, 0.25, 0.12, 0.15], dtype=float)   # final limit of the original curriculum


class TaskScheduleWrapper(gym.Wrapper):
    """Outermost wrapper; set_task() changes only the exogenous task distribution of the underlying env."""

    def set_task(self, targets: Sequence[float], rand_scale: float, hold) -> bool:
        b = self.env.unwrapped
        b.yaw_targets = [float(v) for v in targets]
        b.randomizer.scale = float(np.clip(rand_scale, 0.0, 1.0))
        b.hold_min, b.hold_max = int(hold[0]), int(hold[1])
        return True


def phase_boundaries(total: int, schedule=TASK_SCHEDULE):
    ends, acc = [], 0.0
    for ph in schedule:
        acc += float(ph["frac"]); ends.append(int(round(acc * int(total))))
    ends[-1] = int(total)
    return ends


def _build_env(*, seed, n_envs, max_steps, use_residual, prev_obs_rms=None, first_phase=None):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecNormalize
    from morphing_glider.training import infrastructure as inf
    from morphing_glider.environment.wrappers import ProgressiveTwistWrapper, ResidualHeuristicWrapper
    from morphing_glider.controllers.heuristic import VirtualTendonHeuristicController
    phase_dict = {"name": "task_curriculum", "twist_factor": 1.0, "rand_scale": 1.0, "ramp_steps": 0,
                  "start_twist_factor": 1.0, "reward_shaper": None}

    def thunk(rank):
        def _init():
            # identical to flat SAC: rpl 70 deg, coupling 1, stability weight 0.03, no reward shaping, full twist
            env = inf.make_env(seed=int(seed + rank), domain_rand_scale=1.0, max_steps=int(max_steps), for_eval=False,
                               twist_enabled=True, include_omega_cross=True, roll_pitch_limit_deg=70.0,
                               coupling_scale=1.0, stability_weight=0.03)
            env = Monitor(env)
            if use_residual:
                heur = VirtualTendonHeuristicController(yaw_rate_max=max(abs(v) for v in DEFAULT_YAW_TARGETS))
                env = ResidualHeuristicWrapper(env, heuristic=heur, residual_limit=RESIDUAL_LIMIT,
                                               action_space_limit=RESIDUAL_LIMIT)
            env = ProgressiveTwistWrapper(env, phase=phase_dict, twist_factor=1.0, reward_shaper=None, ramp_steps=0)
            env = TaskScheduleWrapper(env)
            if first_phase is not None:
                env.set_task(first_phase["targets"], first_phase["rand"], first_phase["hold"])
            return env
        return _init

    vec = inf.make_vec_env([thunk(i) for i in range(int(n_envs))], mode=inf.VECENV_MODE,
                           start_method=inf.SUBPROC_START_METHOD)
    vecnorm = None
    if inf.USE_VECNORMALIZE:
        vecnorm = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
        if prev_obs_rms is not None: vecnorm.obs_rms = prev_obs_rms
        vec = vecnorm
        inf.warmup_vecnormalize(vec, n_steps=3500, use_residual_hint=use_residual)
    return vec, vecnorm


def train_task_curriculum_sac(*, total_timesteps, seed, n_envs, max_steps, eval_every_steps, eval_episodes,
                              eval_seed_base, use_residual=False, schedule=TASK_SCHEDULE, callback=None,
                              resume_dir: Optional[str] = None, pause_file: Optional[str] = None):
    from morphing_glider.evaluation import evaluate_controller, summarize_metrics
    from morphing_glider.config import DEVICE
    from morphing_glider.controllers.sb3_controller import SB3Controller
    from morphing_glider.training.infrastructure import build_sac_model_baseline, _standardize_evaltrace_append
    from morphing_glider.training.resume import (PauseRequested, PauseCheckCallback, pause_flag_set,
                                                 save_snapshot, load_snapshot)
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList

    algo = "residual_task_curriculum" if use_residual else "task_curriculum"
    ends = phase_boundaries(total_timesteps, schedule)
    logs: Dict[str, Any] = {"evaltrace": [], "eval_details": [], "phase_boundaries": [],
                            "phases": [dict(p, targets=list(p["targets"]), hold=list(p["hold"]), end_step=e)
                                       for p, e in zip(schedule, ends)],
                            "use_residual": bool(use_residual), "algo_name": algo, "resumes": []}
    pause_cb = PauseCheckCallback(pause_file) if pause_file else None
    cbs = [c for c in (callback, pause_cb) if c is not None]
    learn_cb = CallbackList(cbs) if len(cbs) > 1 else (cbs[0] if cbs else None)
    snap = load_snapshot(resume_dir)
    S: Dict[str, Any] = {"trained": 0, "first": True, "chunk_end": None, "phase": 0}
    vec = vecnorm = model = None

    def phase_of(step):
        return int(min(np.searchsorted(ends, step, side="right"), len(schedule) - 1))

    def apply_phase(k):
        p = schedule[k]
        vec.env_method("set_task", list(p["targets"]), float(p["rand"]), tuple(p["hold"]))
        S["phase"] = int(k)
        print(f"  [TASK] phase {k+1}/{len(schedule)} {p['name']}: refs={list(p['targets'])} rand={p['rand']} "
              f"hold={tuple(p['hold'])} until step {ends[k]}")

    try:
        if snap is None:
            vec, vecnorm = _build_env(seed=seed + 3000, n_envs=n_envs, max_steps=max_steps, use_residual=use_residual,
                                      first_phase=schedule[0])
            model = build_sac_model_baseline(vec, seed=seed, tensorboard_log=f"tb_{algo}")
            apply_phase(0)
        else:
            S.update(snap["state"]); logs = S["logs"]
            vec, vecnorm = _build_env(seed=seed + 3000 + 7919 * (len(logs.get("resumes", [])) + 1), n_envs=n_envs,
                                      max_steps=max_steps, use_residual=use_residual, prev_obs_rms=S.get("obs_rms"),
                                      first_phase=schedule[int(S.get("phase", 0))])
            if vecnorm is not None and S.get("obs_rms") is not None:
                vecnorm.obs_rms = copy.deepcopy(S["obs_rms"])
            model = SAC.load(snap["model"], env=vec, device=DEVICE)
            model.load_replay_buffer(snap["replay"])
            if callback is not None and hasattr(callback, "records"):
                callback.records = list(S.get("diag_records", []))
            logs.setdefault("resumes", []).append({"global_steps": int(model.num_timesteps)})
            apply_phase(phase_of(int(S["trained"])))
            print(f"[RESUME] {algo} at trained={S['trained']} (t={model.num_timesteps})")

        def snapshot(reason):
            if not resume_dir: return
            S["obs_rms"] = copy.deepcopy(vecnorm.obs_rms) if vecnorm is not None else None
            S["logs"] = logs
            if callback is not None and hasattr(callback, "records"): S["diag_records"] = callback.records
            save_snapshot(resume_dir, model=model, state=dict(S))
            print(f"  [SNAPSHOT:{reason}] trained={S['trained']} t={model.num_timesteps}")

        def eval_now(tag):    # identical to the flat-SAC training-time evaluation (target task, rand 1.0)
            ctrl = SB3Controller(model, obs_rms=(vecnorm.obs_rms if vecnorm else None),
                                 clip_obs=(vecnorm.clip_obs if vecnorm else 10.0))
            mets, _ = evaluate_controller(ctrl, n_episodes=int(eval_episodes), eval_seed_base=int(eval_seed_base),
                domain_rand_scale=1.0, max_steps=int(max_steps), twist_factor=1.0, use_residual_env=bool(use_residual),
                residual_limit=(RESIDUAL_LIMIT if use_residual else None), store_histories=False,
                roll_pitch_limit_deg=70.0, coupling_scale=1.0, stability_weight=0.03)
            stats = summarize_metrics(mets, label=f"EVAL({tag}) {algo} | rand=1.0", ci_method="bca",
                                      print_cost_terms=False)
            _standardize_evaltrace_append(logs, tag=str(tag), phase_name=schedule[S["phase"]]["name"],
                                          global_steps=int(model.num_timesteps), stats=stats,
                                          eval_rand_scale=1.0, eval_rpl=70.0)
            return stats

        if snap is None:
            print(f"\n--- TRAIN {algo.upper()} | seed={seed} | envs={n_envs} | steps={total_timesteps} | ends={ends} ---")
            eval_now("init")
        while S["trained"] < int(total_timesteps):
            k = phase_of(int(S["trained"]))
            if k != S["phase"]:
                logs["phase_boundaries"].append({"phase": schedule[S["phase"]]["name"],
                                                 "global_steps": int(model.num_timesteps)})
                apply_phase(k)
            if S.get("chunk_end") is None:
                E = int(eval_every_steps); nxt = (int(S["trained"]) // E + 1) * E     # stay on the eval grid
                S["chunk_end"] = int(min(nxt, ends[k], int(total_timesteps)))
            chunk = int(max(1, S["chunk_end"] - S["trained"]))
            t0 = time.time(); before = int(model.num_timesteps)
            try: model.learn(total_timesteps=chunk, reset_num_timesteps=bool(S["first"]), progress_bar=False, callback=learn_cb)
            except TypeError: model.learn(total_timesteps=chunk, reset_num_timesteps=bool(S["first"]), callback=learn_cb)
            S["first"] = False; S["trained"] += int(model.num_timesteps) - before; wall = time.time() - t0
            logs.setdefault("wall_clock_s", []).append(float(wall))
            if pause_flag_set(pause_file) and (pause_cb is None or pause_cb.paused or S["trained"] < S["chunk_end"]):
                snapshot("pause"); raise PauseRequested(f"paused {algo}")
            S["chunk_end"] = None
            if S["trained"] % int(eval_every_steps) == 0 or S["trained"] >= int(total_timesteps):
                eval_now(str(S["trained"]))
            snapshot("chunk")
            print(f" [TRAIN] {algo}_steps={S['trained']}/{int(total_timesteps)} | phase={schedule[S['phase']]['name']} | wall={wall:.1f}s")
        logs["phase_boundaries"].append({"phase": schedule[S["phase"]]["name"], "global_steps": int(model.num_timesteps)})
        return model, vecnorm, logs
    finally:
        try:
            if vec is not None: vec.close()
        except Exception: pass
        gc.collect()
