"""Episode execution, per-episode metric extraction, and metric summarization."""

import numpy as np

from morphing_glider.config import (
    MIN_EPISODE_SURVIVAL_STEPS,
    REWARD_W_TRACK, REWARD_W_STRUCT, REWARD_SURVIVAL_BONUS,
    SETTLING_REF_MIN_ABS, SETTLING_BAND_MIN, SETTLING_BAND_GAIN,
)
from morphing_glider.utils.numeric import rms, mae, bootstrap_mean_ci_bca
from morphing_glider.utils.quaternion import quat_to_euler_xyz


# ---------------------------------------------------------------------------
# Local 3-return variant matching the original morphing_glider.py signature
# (the package utils/numeric.py version returns only (mean, std)).
# ---------------------------------------------------------------------------
def _finite_mean_std(x):
    a = np.asarray(x, dtype=float); a = a[np.isfinite(a)]
    if a.size == 0: return float("nan"), float("nan"), 0
    return float(np.mean(a)), float(np.std(a, ddof=0)), int(a.size)


# ================================================================
# Episode execution
# ================================================================

def run_episode(env, controller, *, deterministic=True, seed=None, max_steps=None):
    obs, info = env.reset(seed=seed)
    if hasattr(controller, "reset") and callable(getattr(controller, "reset")): controller.reset()
    T = int(env.unwrapped.max_steps if max_steps is None else max_steps); hist = []
    for t in range(T):
        action, _ = controller.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        b = env.unwrapped; roll, pitch, yaw = quat_to_euler_xyz(b.q)
        hist.append({"t": t, "yaw_rate": float(info.get("yaw_rate", b.omega[2])),
                      "yaw_ref": float(info.get("yaw_ref", b.yaw_ref)),
                      "roll": float(info.get("roll", roll)), "pitch": float(info.get("pitch", pitch)),
                      "yaw": float(info.get("yaw", yaw)),
                      "speed": float(info.get("speed", np.linalg.norm(b.vel_world))),
                      "altitude": float(info.get("altitude", b.pos_world[2])),
                      "vz_world": float(info.get("vz_world", b.vel_world[2])),
                      "reward": float(reward), "action": np.array(action, dtype=float).copy(),
                      "terminated": bool(terminated), "truncated": bool(truncated), "info": dict(info)})
        if terminated or truncated: break
    return hist


# ================================================================
# Segment index computation
# ================================================================

def _segment_indices(yaw_ref):
    if yaw_ref.size == 0: return []
    change = np.where(np.abs(np.diff(yaw_ref)) > 1e-12)[0] + 1
    idx = [0] + change.tolist() + [int(yaw_ref.size)]
    segs = [(idx[i], idx[i+1]) for i in range(len(idx)-1) if idx[i+1] > idx[i]]
    return segs if segs else [(0, int(yaw_ref.size))]


# ================================================================
# Per-episode metric extraction
# ================================================================

def compute_episode_metrics(history, *, horizon_T):
    if len(history) == 0: return {"T": 0}
    yr = np.array([h["yaw_rate"] for h in history]); yref = np.array([h["yaw_ref"] for h in history])
    err = yr - yref; roll = np.array([h["roll"] for h in history]); pitch = np.array([h["pitch"] for h in history])
    speed = np.array([h["speed"] for h in history]); alt = np.array([h["altitude"] for h in history])
    vz = np.array([h.get("vz_world", np.nan) for h in history])
    act_norms = []
    for h in history:
        info = h.get("info", {})
        if isinstance(info, dict) and "total_action_norm" in info and np.isfinite(info["total_action_norm"]):
            act_norms.append(float(info["total_action_norm"]))
        else: act_norms.append(float(np.linalg.norm(h["action"])))
    act = np.array(act_norms); rew = np.array([h["reward"] for h in history])
    terminated = bool(history[-1].get("terminated", False)); truncated = bool(history[-1].get("truncated", False))
    reason = str(history[-1].get("info", {}).get("termination_reason", ""))
    failure = terminated and not truncated
    FAIL_ERR = 1.0
    if int(horizon_T) <= 0: rms_h = rms(err)
    else:
        pad = int(max(0, int(horizon_T) - int(err.size)))
        err_h = np.concatenate([err, np.full((pad,), FAIL_ERR)]) if pad > 0 else err[:int(horizon_T)]
        rms_h = rms(err_h)
    segs = _segment_indices(yref); steady_err = []; transient_err = []; settle_times = []
    for (s, e) in segs:
        seg_err = err[s:e]; seg_ref = float(yref[s])
        if seg_err.size < 4: continue
        k = int(np.clip(max(8, int(0.20*(e-s))), 8, max(8, (e-s)//2)))
        transient_err.append(seg_err[:k]); steady_err.append(seg_err[k:])
        if abs(seg_ref) < SETTLING_REF_MIN_ABS: continue
        band = max(SETTLING_BAND_MIN, SETTLING_BAND_GAIN*abs(seg_ref))
        st = float("nan")
        if seg_err.size >= 6:
            for i in range(seg_err.size):
                if np.all(np.abs(seg_err[i:]) <= band): st = float(i); break
        settle_times.append(st)
    se = np.concatenate(steady_err) if steady_err else np.array([])
    te = np.concatenate(transient_err) if transient_err else np.array([])
    sts = np.asarray(settle_times); sts = sts[np.isfinite(sts)]
    pl = np.array([h.get("info",{}).get("power_loss_total",np.nan) for h in history]); pl = pl[np.isfinite(pl)]
    def mi(key): v = np.array([h.get("info",{}).get(key,np.nan) for h in history]); v=v[np.isfinite(v)]; return float(np.mean(v)) if v.size else float("nan")
    vz_f = vz[np.isfinite(vz)]
    return {
        "T": int(len(history)), "ended_early": float(len(history)<int(horizon_T)), "failure": float(failure),
        "term_attitude": float(reason=="attitude_limit"), "term_stall": float(reason=="stall"),
        "term_ground": float(reason=="ground"), "term_nan": float(reason=="nan"),
        "term_other": float(reason!="" and reason not in ("attitude_limit","stall","ground","nan")),
        "rms_yaw": rms(err), "mae_yaw": mae(err), "rms_yaw_horizon": float(rms_h),
        "rms_yaw_transient": rms(te) if te.size else float("nan"),
        "rms_yaw_steady": rms(se) if se.size else float("nan"),
        "mean_settle_time": float(np.mean(sts)) if sts.size else float("nan"),
        "rms_roll": rms(roll), "rms_pitch": rms(pitch),
        "mean_speed": float(np.mean(speed)) if speed.size else float("nan"),
        "min_speed": float(np.min(speed)) if speed.size else float("nan"),
        "mean_vz": float(np.mean(vz_f)) if vz_f.size else float("nan"),
        "min_altitude": float(np.min(alt)) if alt.size else float("nan"),
        "mean_altitude": float(np.mean(alt)) if alt.size else float("nan"),
        "delta_altitude": float(alt[-1]-alt[0]) if alt.size else float("nan"),
        "mean_action_norm": float(np.mean(act)) if act.size else float("nan"),
        "total_reward": float(np.sum(rew)) if rew.size else float("nan"),
        "mean_power_loss": float(np.mean(pl)) if pl.size else float("nan"),
        "mean_cost_track": mi("cost_track"), "mean_cost_att": mi("cost_att"),
        "mean_cost_rates": mi("cost_rates"), "mean_cost_ctrl": mi("cost_ctrl"),
        "mean_cost_jerk": mi("cost_jerk"), "mean_cost_power": mi("cost_power"),
        "mean_cost_struct": mi("cost_struct"), "mean_cost_zsym": mi("cost_zsym"),
        "mean_total_cost": mi("total_cost"),
    }


# ================================================================
# Multi-episode summarization
# ================================================================

def summarize_metrics(metrics, *, label, ci_method="bca", ci=95.0, print_cost_terms=True):
    def arr(key): return np.array([m.get(key, float("nan")) for m in metrics])
    out = {"label": label, "n_episodes": int(len(metrics))}
    print("\n" + "="*80); print(f"SUMMARY :: {label}"); print("="*80)
    print(f"Episodes: {len(metrics)}")
    fail = arr("failure"); fail_rate = float(np.nanmean(fail)) if np.isfinite(fail).any() else float("nan")
    out["failure_rate"] = fail_rate
    print(f"Failure rate: {fail_rate*100:.1f}%")

    # Failure mode breakdown
    term_att = arr("term_attitude"); term_stall = arr("term_stall")
    term_gnd = arr("term_ground"); term_nan = arr("term_nan")
    n_att = int(np.nansum(term_att)) if np.isfinite(term_att).any() else 0
    n_stall = int(np.nansum(term_stall)) if np.isfinite(term_stall).any() else 0
    n_gnd = int(np.nansum(term_gnd)) if np.isfinite(term_gnd).any() else 0
    n_nan = int(np.nansum(term_nan)) if np.isfinite(term_nan).any() else 0
    if n_att + n_stall + n_gnd + n_nan > 0:
        print(f"  Failure modes: attitude={n_att}, stall={n_stall}, ground={n_gnd}, nan={n_nan}")

    x = arr("rms_yaw_horizon"); mean, lo, hi = bootstrap_mean_ci_bca(x, ci=ci, seed=123)
    m_raw, s_raw, n_raw = _finite_mean_std(x)
    print(f"Yaw RMS@H: {m_raw:.4f} ± {s_raw:.4f} CI[{lo:.4f}, {hi:.4f}] (n={n_raw})")
    out.update({"mean_rmsh": float(mean), "lo_rmsh": float(lo), "hi_rmsh": float(hi), "std_rmsh": float(s_raw), "n_rmsh": float(n_raw)})

    se_valid = [float(m.get("rms_yaw_steady", np.nan)) for m in metrics
                if int(m.get("T",0)) >= MIN_EPISODE_SURVIVAL_STEPS and np.isfinite(m.get("rms_yaw_steady", np.nan))]
    hirmssteady = float(np.percentile(se_valid, 85)) if se_valid else 999.0
    hirms_p95 = float(np.percentile(se_valid, 95)) if se_valid else 999.0
    hirms_max = float(np.max(se_valid)) if se_valid else 999.0
    out["hirmssteady"] = hirmssteady
    out["hirms_p95"] = hirms_p95
    out["hirms_max"] = hirms_max
    print(f"Gate hirmssteady(p85): {hirmssteady:.4f} (n_valid={len(se_valid)})")
    print(f"  hirms_p95: {hirms_p95:.4f}, hirms_max: {hirms_max:.4f}")

    for k in ["rms_yaw_steady", "rms_yaw_transient", "mean_settle_time", "mean_action_norm",
              "mean_speed", "mean_altitude", "delta_altitude", "mean_power_loss"]:
        a = arr(k); mm, ss, nn = _finite_mean_std(a)
        if nn > 0:
            mci, lci, hci = bootstrap_mean_ci_bca(a, ci=ci, seed=456+(hash(k)%997))
            out[f"mean_{k}"] = float(mci); out[f"lo_{k}"] = float(lci); out[f"hi_{k}"] = float(hci)
        else:
            out[f"mean_{k}"] = float("nan"); out[f"lo_{k}"] = float("nan"); out[f"hi_{k}"] = float("nan")

    if print_cost_terms:
        for ck in ["mean_cost_track","mean_cost_att","mean_cost_rates","mean_cost_ctrl",
                    "mean_cost_jerk","mean_cost_power","mean_cost_struct","mean_cost_zsym","mean_total_cost"]:
            a = arr(ck); mm,ss,nn = _finite_mean_std(a)
            if nn > 0: print(f"  {ck}: {mm:.6f} ± {ss:.6f}")

    track_m = arr("mean_cost_track"); struct_m = arr("mean_cost_struct")
    tr_m = arr("tracking_reward") if "tracking_reward" in (metrics[0] if metrics else {}) else np.array([])
    if np.isfinite(track_m).any() and np.isfinite(struct_m).any():
        wt = REWARD_W_TRACK*float(np.nanmean(track_m)); ws = REWARD_W_STRUCT*float(np.nanmean(struct_m))
        print(f"[Weighted check] w_track*cost={wt:.6f} vs w_struct*cost={ws:.6f} {'OK' if ws<wt else 'WARN'}")
    if tr_m.size and np.isfinite(tr_m).any():
        print(f"[Reward check] mean tracking_reward={float(np.nanmean(tr_m)):.4f}, survival_bonus={REWARD_SURVIVAL_BONUS:.2f}")
    return out
