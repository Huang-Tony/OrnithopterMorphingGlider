#!/usr/bin/env python3
"""Policy-interpretability suite (revision R1). Every number is saved to analysis/interpretability.json.

Lens 1  Morphing strategy: steady-state use of the six virtual morphing inputs versus the yaw reference
        (linear fit, R^2, usage), for baseline and curriculum SAC (all seeds) - which channels make the yaw.
Lens 2  Input sensitivity: Jacobian of the deterministic action with respect to the NORMALIZED observation
        (effect of a one-standard-deviation change), averaged over 500 visited states; the raw-unit variant
        used in the submitted manuscript is also reported to show its scale artifact.
Lens 3  Latent structure: PCA of the actor's last hidden layer on visited states; correlation of the
        principal components with reference, roll and yaw error; fraction of time per flight regime.
Lens 4  Distillation: DAgger distillation of the SAC teacher into a Kolmogorov-Arnold network (KAN); the
        student is evaluated with the standard protocol (full data) and its inference cost is measured.
Lens 5  Cross-agent comparison: Lens 1 for curriculum versus baseline SAC.
The teacher is the baseline seed with the median nominal RMS@H (no cherry-picking).
"""

import glob
import json
import os
import time

import numpy as np

import common  # noqa: F401
from common import QUICK, RUNS_DIR, MAX_STEPS_EP, FINAL_EVAL_EPS, EVAL_SEED_BASE, CONDITIONS, TRAIN_SEEDS, algo_seed, dump_json
from evaluate import evaluate, load_sac

EV = os.path.join(RUNS_DIR, "eval_final"); OUTJ = os.path.join(RUNS_DIR, "analysis", "interpretability.json")
NAMES = ["diff_sweep", "sym_sweep", "diff_dihedral", "sym_dihedral", "diff_span", "sym_span"]
GROUPS = {"attitude": range(0, 6), "body rates": range(6, 9), "air data": range(9, 13), "altitude/climb": range(13, 15),
          "reference": range(15, 17), "tip positions": range(17, 29), "spar shape": range(29, 41)}


def virtual(a):
    a = np.asarray(a)
    return np.stack([(a[..., 0] - a[..., 3]) / 2, (a[..., 0] + a[..., 3]) / 2, (a[..., 2] - a[..., 5]) / 2,
                     (a[..., 2] + a[..., 5]) / 2, (a[..., 1] + a[..., 4]) / 2, (a[..., 1] - a[..., 4]) / 2], axis=-1)


def steady_mask(yref):
    m = np.isfinite(yref); out = np.zeros_like(m)
    for e in range(yref.shape[0]):
        y = yref[e]; ok = np.isfinite(y); n = int(ok.sum())
        if n < 10: continue
        ch = np.r_[0, np.where(np.abs(np.diff(y[:n])) > 1e-9)[0] + 1, n]
        for s, t in zip(ch[:-1], ch[1:]):
            k = int(np.clip(max(8, int(0.2 * (t - s))), 8, max(8, (t - s) // 2)))
            out[e, s + k:t] = True
    return out & m


def strategy(algo):
    rows = {}
    for s in TRAIN_SEEDS:
        f = os.path.join(EV, f"sac_{algo}_seed{algo_seed(algo, s)}_nominal.npz")
        if not os.path.exists(f): continue
        d = np.load(f); V = virtual(d["action"]); yr = d["yaw_ref"]; m = steady_mask(yr)
        for j, nm in enumerate(NAMES):
            x, y = yr[m], V[..., j][m]
            p = np.polyfit(x, y, 1); r2 = 1 - np.var(y - np.polyval(p, x)) / max(np.var(y), 1e-12)
            rows.setdefault(nm, []).append({"slope": float(p[0]), "intercept": float(p[1]), "r2": float(r2),
                                            "rms_use": float(np.sqrt(np.mean(y ** 2))),
                                            "by_ref": {f"{v:+.1f}": float(np.mean(y[np.isclose(x, v)])) for v in (-0.6, -0.3, 0.0, 0.3, 0.6)
                                                       if np.any(np.isclose(x, v))}})
    return {nm: {"per_seed": v, "slope_mean": float(np.mean([r["slope"] for r in v])),
                 "r2_mean": float(np.mean([r["r2"] for r in v])), "rms_use_mean": float(np.mean([r["rms_use"] for r in v]))}
            for nm, v in rows.items()}


def teacher_seed():
    best = []
    for s in TRAIN_SEEDS:
        f = os.path.join(EV, f"sac_baseline_seed{s}_nominal.json")
        if os.path.exists(f):
            best.append((np.mean([e["rms_yaw_horizon"] for e in json.load(open(f))["episodes"]]), s))
    best.sort(); return best[len(best) // 2][1], best


def actor_mean(model, obs_n):
    import torch
    with torch.no_grad():
        mu, _, _ = model.actor.get_action_dist_params(torch.as_tensor(obs_n, dtype=torch.float32))
        return np.tanh(mu.numpy())


def sensitivity_and_latent(seed):
    import torch
    ctrl = load_sac(os.path.join(RUNS_DIR, "models", f"baseline_seed{seed}.zip"))
    sb3 = ctrl.ctrl; model = sb3.model
    d = np.load(os.path.join(EV, f"sac_baseline_seed{seed}_nominal.npz"))
    O = d["obs"].reshape(-1, 41); keep = np.isfinite(O).all(1)
    meta = {k: d[k].reshape(-1)[keep] for k in ("yaw_ref", "roll", "pitch", "yaw_rate")}
    O = O[keep]; rng = np.random.default_rng(0); idx = rng.choice(O.shape[0], size=min(500, O.shape[0]), replace=False)
    S = O[idx]; mean, var = np.asarray(sb3.obs_rms.mean), np.asarray(sb3.obs_rms.var); std = np.sqrt(var + 1e-8)
    Sn = np.clip((S - mean) / std, -10, 10)
    eps = 0.05; J = np.zeros((41,))
    for j in range(41):
        P = Sn.copy(); P[:, j] += eps; M = Sn.copy(); M[:, j] -= eps
        J[j] = float(np.mean(np.abs(actor_mean(model, P) - actor_mean(model, M)) / (2 * eps)))
    # raw-unit variant (submitted manuscript): fixed step of 1e-3 in raw units
    Jr = np.zeros(41)
    for j in range(41):
        P = S.copy(); P[:, j] += 1e-3
        Pn = np.clip((P - mean) / std, -10, 10)
        Jr[j] = float(np.mean(np.abs(actor_mean(model, Pn) - actor_mean(model, Sn)) / 1e-3))
    from morphing_glider.environment.observation import OBS_IDX
    names = {v: k for k, v in OBS_IDX.items()}
    imp = {names[j]: float(J[j]) for j in range(41)}; imp_raw = {names[j]: float(Jr[j]) for j in range(41)}
    groups = {g: float(np.sum(J[list(r)])) for g, r in GROUPS.items()}; tot = sum(groups.values())
    groups = {g: v / tot for g, v in groups.items()}
    # latent lens: last hidden layer of the actor
    acts = []
    def hook(_m, _i, o): acts.append(o.detach().numpy())
    layers = [m for m in model.actor.latent_pi.modules() if isinstance(m, torch.nn.ReLU)]
    h = layers[-1].register_forward_hook(hook)
    Lidx = rng.choice(O.shape[0], size=min(4000, O.shape[0]), replace=False)
    _ = actor_mean(model, np.clip((O[Lidx] - mean) / std, -10, 10)); h.remove()
    Z = acts[-1]; Zc = Z - Z.mean(0); U, sv, Vt = np.linalg.svd(Zc, full_matrices=False)
    ev = (sv ** 2) / np.sum(sv ** 2); pcs = Zc @ Vt[:2].T
    yr = meta["yaw_ref"][Lidx]; roll = meta["roll"][Lidx]; pitch = meta["pitch"][Lidx]; err = meta["yaw_rate"][Lidx] - yr
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    regime = np.where((np.abs(roll) > 0.6) | (np.abs(pitch) > 0.6), "stressed",
             np.where((np.abs(roll) > 0.3) | (np.abs(pitch) > 0.3), "large attitude",
             np.where((np.abs(yr) > 0.4) & (np.abs(err) < 0.15), "turning on target",
             np.where(np.abs(err) > 0.2, "correcting error", "steady glide"))))
    frac = {r: float(np.mean(regime == r)) for r in np.unique(regime)}
    dead = float(np.mean(Z.max(0) <= 1e-9))
    latent = {"explained_variance": ev[:5].tolist(), "pc1_corr": {"yaw_ref": corr(pcs[:, 0], yr), "roll": corr(pcs[:, 0], roll), "yaw_error": corr(pcs[:, 0], err)},
              "pc2_corr": {"yaw_ref": corr(pcs[:, 1], yr), "roll": corr(pcs[:, 1], roll), "yaw_error": corr(pcs[:, 1], err)},
              "regime_fractions": frac, "dead_units_fraction": dead, "n_states": int(Z.shape[0])}
    np.savez_compressed(os.path.join(RUNS_DIR, "analysis", "latent_pcs.npz"), pcs=pcs, yaw_ref=yr, roll=roll, err=err, regime=regime)
    return {"normalized": imp, "raw_unit": imp_raw, "groups": groups, "n_states": int(S.shape[0]), "eps_normalized": eps}, latent


def kan_distillation(seed, out):
    import torch
    torch.set_num_threads(4)
    from morphing_glider.interpretability.kan import KANPolicyNetwork
    from morphing_glider.interpretability.dagger import DAggerDistillation, _NormalizedStudentWrapper
    ctrl = load_sac(os.path.join(RUNS_DIR, "models", f"baseline_seed{seed}.zip")); sb3 = ctrl.ctrl
    torch.manual_seed(0); np.random.seed(0)
    kan = KANPolicyNetwork(obs_dim=41, action_dim=6, hidden_dim=64, n_bases=10)
    dag = DAggerDistillation(expert=sb3, student=kan, n_iterations=2 if QUICK else 15, episodes_per_iter=1 if QUICK else 8, max_steps=MAX_STEPS_EP,
                             mix_probability=0.9, beta_decay=0.8, learning_rate=5e-4, obs_rms=sb3.obs_rms, clip_obs=sb3.clip_obs)
    t0 = time.time(); hist = dag.train(verbose=True); wall = time.time() - t0
    torch.save(kan.state_dict(), os.path.join(RUNS_DIR, "models", "kan_student.pt"))
    student = _NormalizedStudentWrapper(kan, sb3.obs_rms, sb3.clip_obs)
    for cond, rs in CONDITIONS.items():
        evaluate(student, label="KAN student", condition=cond, rand_scale=rs, n_episodes=FINAL_EVAL_EPS, seed_base=EVAL_SEED_BASE,
                 out_dir=EV, tag=f"kan_student_{cond}", verbose=True)
    # inference cost (CPU, single observation)
    o = np.zeros(41, dtype=np.float32)
    def lat(fn, n=2000):
        for _ in range(50): fn(o)
        t = time.perf_counter()
        for _ in range(n): fn(o)
        return 1e6 * (time.perf_counter() - t) / n
    n_kan = int(sum(p.numel() for p in kan.parameters()))
    n_actor = int(sum(p.numel() for n_, p in sb3.model.actor.named_parameters() if not n_.startswith("log_std")))
    out["kan"] = {"history": {k: [float(x) for x in v] for k, v in hist.items() if isinstance(v, list)},
                  "params": n_kan, "teacher_actor_params_mean_path": n_actor, "compression": n_actor / n_kan,
                  "train_wall_s": wall, "latency_us_kan": lat(lambda x: student.predict(x)),
                  "latency_us_sac": lat(lambda x: sb3.predict(x))}


def main():
    out = json.load(open(OUTJ)) if os.path.exists(OUTJ) else {}
    seed, ranking = teacher_seed(); out["teacher_seed"] = seed; out["teacher_ranking"] = ranking
    print("teacher seed", seed, ranking)
    out["strategy_baseline"] = strategy("baseline"); out["strategy_curriculum"] = strategy("curriculum")
    out["sensitivity"], out["latent"] = sensitivity_and_latent(seed)
    dump_json(out, OUTJ)
    if "kan" not in out:
        kan_distillation(seed, out); dump_json(out, OUTJ)
    top = sorted(out["sensitivity"]["normalized"].items(), key=lambda kv: -kv[1])[:5]
    print("top normalized features:", top); print("groups:", out["sensitivity"]["groups"])
    print("latent:", out["latent"]["explained_variance"][:3], out["latent"]["regime_fractions"])
    print("kan:", {k: v for k, v in out["kan"].items() if k != "history"})


if __name__ == "__main__":
    main()
