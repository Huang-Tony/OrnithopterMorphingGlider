"""Reward computer and monitoring for the morphing glider environment."""

import math
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np

from morphing_glider.config import (
    REWARD_W_TRACK, REWARD_W_ATT_GAIN, REWARD_W_ATT_FLOOR,
    REWARD_W_RATES_GAIN, REWARD_W_RATES_FLOOR,
    MAX_COST_ATT_REF, MAX_COST_RATES_REF,
    REWARD_W_CTRL, REWARD_W_JERK, REWARD_W_POWER, REWARD_W_STRUCT, REWARD_W_ZSYM,
    REWARD_CLIP_MIN, REWARD_CLIP_MAX, REWARD_SURVIVAL_BONUS,
    REWARD_TRACKING_SHARPNESS, REWARD_W_WALL, REWARD_WALL_MARGIN,
)


class RewardComputer:
    r"""Computes the morphing glider reward with positive tracking + survival bonus.

    Reward equation:
        r = survival_bonus
          + w_track * exp(-sharpness * e_yaw^2)     [positive tracking reward]
          - w_att * (roll^2 + pitch^2) / ref_att     [attitude penalty]
          - w_rates * (wp^2 + wq^2) / ref_rates     [angular rate penalty]
          - w_ctrl * ||u||^2                         [control effort]
          - w_jerk * ||du||^2                        [actuator smoothness]
          - w_power * P_loss                         [power consumption]
          - w_struct * E_struct                       [structural stress]
          - w_zsym * z_sym^2                         [asymmetry penalty]
          - w_wall * soft_wall(roll, pitch)          [soft attitude limit]

    References:
        [LILLICRAP_2016] Continuous control with deep RL.
        [NG_1999] Policy invariance under reward transformations.
    """
    def __init__(self, *, w_track=REWARD_W_TRACK,
                 w_att_gain=REWARD_W_ATT_GAIN, w_att_floor=REWARD_W_ATT_FLOOR,
                 w_rates_gain=REWARD_W_RATES_GAIN, w_rates_floor=REWARD_W_RATES_FLOOR,
                 max_cost_att_ref=MAX_COST_ATT_REF, max_cost_rates_ref=MAX_COST_RATES_REF,
                 w_ctrl=REWARD_W_CTRL, w_jerk=REWARD_W_JERK,
                 w_power=REWARD_W_POWER, w_struct=REWARD_W_STRUCT,
                 w_zsym=REWARD_W_ZSYM,
                 clip_min=REWARD_CLIP_MIN, clip_max=REWARD_CLIP_MAX,
                 survival_bonus=REWARD_SURVIVAL_BONUS,
                 tracking_sharpness=REWARD_TRACKING_SHARPNESS,
                 w_wall=REWARD_W_WALL,
                 wall_margin=REWARD_WALL_MARGIN):
        self.w_track = w_track; self.w_att_gain = w_att_gain; self.w_att_floor = w_att_floor
        self.w_rates_gain = w_rates_gain; self.w_rates_floor = w_rates_floor
        self.max_cost_att_ref = max_cost_att_ref; self.max_cost_rates_ref = max_cost_rates_ref
        self.w_ctrl = w_ctrl; self.w_jerk = w_jerk; self.w_power = w_power
        self.w_struct = w_struct; self.w_zsym = w_zsym
        self.clip_min = clip_min; self.clip_max = clip_max
        self.survival_bonus = float(survival_bonus)
        self.tracking_sharpness = float(tracking_sharpness)
        self.w_wall = float(w_wall)
        self.wall_margin = float(wall_margin)

    def compute(self, *, yaw_error, roll, pitch,
                omega_p_clipped, omega_q_clipped,
                action, prev_action,
                power_norm, e_sum_norm, z_sym,
                stability_weight,
                roll_pitch_limit=1.22):
        w_att = max(stability_weight * self.w_att_gain, self.w_att_floor) / max(1e-9, self.max_cost_att_ref)
        w_rates = max(stability_weight * self.w_rates_gain, self.w_rates_floor) / max(1e-9, self.max_cost_rates_ref)

        cost_track = float(yaw_error ** 2)
        cost_att = float(roll ** 2 + pitch ** 2)
        cost_rates = float(omega_p_clipped ** 2 + omega_q_clipped ** 2)
        cost_ctrl = float(np.linalg.norm(action) ** 2)
        cost_jerk = float(np.linalg.norm(action - prev_action) ** 2)
        cost_power = float(power_norm)
        cost_struct = float(np.clip(e_sum_norm, 0.0, 2.0))
        cost_zsym = float(z_sym ** 2)

        r_survival = float(self.survival_bonus)
        r_tracking = float(self.w_track * math.exp(
            -self.tracking_sharpness * min(cost_track, 10.0)))

        penalty_att = float(w_att * cost_att)
        penalty_rates = float(w_rates * cost_rates)
        penalty_ctrl = float(self.w_ctrl * cost_ctrl)
        penalty_jerk = float(self.w_jerk * cost_jerk)
        penalty_power = float(self.w_power * cost_power)
        penalty_struct = float(self.w_struct * cost_struct)
        penalty_zsym = float(self.w_zsym * cost_zsym)

        rpl = float(max(roll_pitch_limit, 0.1))
        roll_frac = float(abs(roll)) / rpl
        pitch_frac = float(abs(pitch)) / rpl
        wm = float(self.wall_margin)
        wall_roll = float(max(0.0, math.exp(4.0 * (roll_frac - wm)) - 1.0))
        wall_pitch = float(max(0.0, math.exp(4.0 * (pitch_frac - wm)) - 1.0))
        wall_cost = float(self.w_wall * (wall_roll + wall_pitch))

        total_penalty = (penalty_att + penalty_rates + penalty_ctrl + penalty_jerk
                         + penalty_power + penalty_struct + penalty_zsym + wall_cost)
        total_reward = r_survival + r_tracking - total_penalty
        reward = float(np.clip(total_reward, self.clip_min, self.clip_max))
        total_cost = float(self.w_track * cost_track + total_penalty)

        breakdown = {
            "cost_track": cost_track, "cost_att": cost_att, "cost_rates": cost_rates,
            "cost_ctrl": cost_ctrl, "cost_jerk": cost_jerk, "cost_power": cost_power,
            "cost_struct": cost_struct, "cost_zsym": cost_zsym,
            "tracking_reward": r_tracking, "survival_bonus": r_survival,
            "wall_cost": wall_cost, "total_penalty": total_penalty,
            "total_cost": total_cost, "total_reward": total_reward,
            "w_att_eff": w_att, "w_rates_eff": w_rates,
        }
        return reward, breakdown


class RewardTermMonitor:
    """Accumulates reward breakdown across an episode and logs statistics."""
    def __init__(self):
        self._terms: Dict[str, List[float]] = {}

    def reset(self):
        self._terms = {}

    def update(self, breakdown):
        for k, v in breakdown.items():
            self._terms.setdefault(k, []).append(float(v))

    def summarize(self):
        out = {}
        for k, vals in self._terms.items():
            a = np.array(vals, dtype=float); a = a[np.isfinite(a)]
            if a.size == 0:
                out[k] = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
            else:
                out[k] = {"mean": float(np.mean(a)), "std": float(np.std(a)),
                          "min": float(np.min(a)), "max": float(np.max(a))}
        return out

    def print_table(self, label=""):
        s = self.summarize()
        print(f"\n[RewardTermMonitor] {label}")
        print(f"  {'Term':<15s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
        for k in sorted(s.keys()):
            v = s[k]
            print(f"  {k:<15s} {v['mean']:10.5f} {v['std']:10.5f} {v['min']:10.5f} {v['max']:10.5f}")


def check_reward_term_magnitudes(breakdown_stats):
    """Warn if any reward penalty term contributes <1% or >80% of total penalty."""
    tp = breakdown_stats.get("total_penalty", {}).get("mean", float("nan"))
    if not np.isfinite(tp) or abs(tp) < 1e-12:
        tp = breakdown_stats.get("total_cost", {}).get("mean", float("nan"))
    if not np.isfinite(tp) or abs(tp) < 1e-12:
        return
    cost_keys = ["cost_att", "cost_rates", "cost_ctrl", "cost_jerk",
                 "cost_power", "cost_struct", "cost_zsym", "wall_cost"]
    for k in cost_keys:
        v = breakdown_stats.get(k, {}).get("mean", float("nan"))
        if not np.isfinite(v):
            continue
        frac = abs(v) / (abs(tp) + 1e-12)
        if frac < 0.005:
            warnings.warn(f"Reward term '{k}' contributes <0.5% ({frac*100:.2f}%) of total penalty.", RuntimeWarning)
        if frac > 0.85:
            warnings.warn(f"Reward term '{k}' contributes >85% ({frac*100:.1f}%) of total penalty.", RuntimeWarning)
    tr = breakdown_stats.get("tracking_reward", {}).get("mean", float("nan"))
    if np.isfinite(tr) and tr < 0.1:
        warnings.warn(f"Tracking reward is very low ({tr:.4f}). Agent may not be learning to track.", RuntimeWarning)
