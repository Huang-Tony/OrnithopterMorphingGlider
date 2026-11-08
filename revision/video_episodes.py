#!/usr/bin/env python3
"""Fly the policies shown in the supplementary videos on ONE common evaluation episode and save full state.

Episode: nominal condition, evaluation seed EVAL_SEED_BASE + i, where i is the episode of the teacher SAC policy
(baseline seed with the median nominal RMS@H) whose RMS@H is closest to that policy's median and whose reference
sequence contains both turn directions (representative, not best-case). Policies:
  sac_teacher, sac_seed7_<steps> (training checkpoints), classical_<best>, kan_student (if available).
Output: revision_runs/videos/data/<tag>.npz/.json and episode.json (choice and its rationale).
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402
from common import RUNS_DIR, EVAL_SEED_BASE, TRAIN_SEEDS, dump_json  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.join(RUNS_DIR, "videos", "data")
CKPT_STEPS = [0, 50_000, 150_000, 300_000, 600_000, 1_000_000]


def eps(path):
    return json.load(open(path))["episodes"]


def choose_episode():
    rms = {}
    for s in TRAIN_SEEDS:
        p = os.path.join(RUNS_DIR, "eval_final", f"sac_baseline_seed{s}_nominal.json")
        if os.path.exists(p): rms[s] = float(np.median([e["rms_yaw_horizon"] for e in eps(p)]))
    teacher = sorted(rms, key=rms.get)[len(rms) // 2]
    ij = os.path.join(RUNS_DIR, "analysis", "interpretability.json")
    if os.path.exists(ij):                  # use exactly the teacher of the KAN distillation
        teacher = int(json.load(open(ij)).get("teacher_seed", teacher))
    E = eps(os.path.join(RUNS_DIR, "eval_final", f"sac_baseline_seed{teacher}_nominal.json"))
    arr = np.load(os.path.join(RUNS_DIR, "eval_final", f"sac_baseline_seed{teacher}_nominal.npz"))
    med = np.median([e["rms_yaw_horizon"] for e in E])
    cand = []
    for i, e in enumerate(E):
        ref = arr["yaw_ref"][i]; ref = ref[np.isfinite(ref)]
        if e["failure"] or not (np.any(ref > 0.1) and np.any(ref < -0.1)): continue
        cand.append((abs(e["rms_yaw_horizon"] - med), i))
    i = min(cand)[1] if cand else int(np.argsort([e["rms_yaw_horizon"] for e in E])[len(E) // 2])
    return teacher, i, float(E[i]["rms_yaw_horizon"]), float(med)


def main():
    import torch; torch.set_num_threads(2)
    from evaluate import evaluate, load_sac
    from classical import make_classical
    os.makedirs(OUT, exist_ok=True)
    teacher, i, r_i, med = choose_episode()
    seed = EVAL_SEED_BASE + i
    dump_json({"teacher_seed": teacher, "episode_index": i, "eval_seed": seed, "teacher_rms_h": r_i,
               "teacher_median_rms_h": med, "rule": __doc__}, os.path.join(OUT, "episode.json"))
    pol = {"sac_teacher": load_sac(os.path.join(RUNS_DIR, "models", f"baseline_seed{teacher}.zip"))}
    sj = os.path.join(RUNS_DIR, "analysis", "smooth_study.json")      # smoothness fine-tuned teacher (smooth_study.py)
    if os.path.exists(sj) and json.load(open(sj)).get("selected"):
        sp = os.path.join(RUNS_DIR, "smooth", "models", f"smooth_seed{teacher}_{json.load(open(sj))['selected']}.zip")
        if os.path.exists(sp): pol["sac_smooth"] = load_sac(sp)
    for st in CKPT_STEPS:
        p = os.path.join(RUNS_DIR, "checkpoints", "baseline_seed7", f"step_{st:07d}.zip")
        if st == 1_000_000 and not os.path.exists(p): p = os.path.join(RUNS_DIR, "models", "baseline_seed7.zip")
        if os.path.exists(p): pol[f"sac_seed7_{st:07d}"] = load_sac(p)
    tun = json.load(open(os.path.join(RUNS_DIR, "analysis", "classical_tuning.json")))
    rk = sorted((np.mean([e["rms_yaw_horizon"] for e in eps(f)]), json.load(open(f))["label"])
                for f in glob.glob(os.path.join(RUNS_DIR, "eval_final", "classical_*_nominal.json")))
    best = rk[0][1]
    pol[f"classical_{best}"] = make_classical(best, **tun[best]["selected"])
    kan_path = os.path.join(RUNS_DIR, "models", "kan_student.pt")
    if os.path.exists(kan_path):          # student distilled from the teacher (interpretability.py), same normalization
        from morphing_glider.interpretability.kan import KANPolicyNetwork
        from morphing_glider.interpretability.dagger import _NormalizedStudentWrapper
        kan = KANPolicyNetwork(obs_dim=41, action_dim=6, hidden_dim=64, n_bases=10)
        kan.load_state_dict(torch.load(kan_path, map_location="cpu")); kan.eval()
        sb3 = pol["sac_teacher"].ctrl
        pol["kan_student"] = _NormalizedStudentWrapper(kan, sb3.obs_rms, sb3.clip_obs)
    for tag, ctrl in pol.items():
        evaluate(ctrl, label=tag, condition="nominal", rand_scale=0.0, n_episodes=1, seed_base=seed,
                 out_dir=OUT, tag=tag, verbose=True)
    print("[video_episodes] teacher seed", teacher, "episode", i, "best classical", best)


if __name__ == "__main__":
    main()
