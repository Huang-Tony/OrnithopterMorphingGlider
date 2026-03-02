"""Numeric helpers and statistics: RMS, bootstrap CI, paired tests."""

import math
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np

from morphing_glider.config import BOOTSTRAP_N_PCT, BOOTSTRAP_N_BCA

try:
    import scipy.stats as spstats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    spstats = None


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0


def mae(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(np.abs(x))) if x.size else 0.0


def finite_mean_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x, ddof=0))


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p):
    if p <= 0.0:
        return -6.0
    if p >= 1.0:
        return 6.0
    t = math.sqrt(-2.0 * math.log(min(p, 1 - p)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
    return x if p > 0.5 else -x


def bootstrap_mean_ci_percentile(x, *, ci=95.0, n_boot=None, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        m = float(np.mean(x)) if x.size else float("nan")
        return m, m, m
    n_boot = n_boot or BOOTSTRAP_N_PCT
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(x, size=x.size, replace=True)) for _ in range(n_boot)])
    alpha = (100.0 - ci) / 2.0
    return float(np.mean(x)), float(np.percentile(means, alpha)), float(np.percentile(means, 100 - alpha))


def bootstrap_mean_ci_bca(x, *, ci=95.0, n_boot=None, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        m = float(np.mean(x)) if x.size else float("nan")
        return m, m, m
    n_boot = n_boot or BOOTSTRAP_N_BCA
    rng = np.random.default_rng(seed)
    theta_hat = float(np.mean(x))
    boot_means = np.array([float(np.mean(rng.choice(x, size=x.size, replace=True))) for _ in range(n_boot)])
    # Bias correction
    z0 = _norm_ppf(float(np.mean(boot_means < theta_hat)))
    # Acceleration (jackknife)
    n = x.size
    jack = np.array([float(np.mean(np.delete(x, i))) for i in range(n)])
    jack_mean = float(np.mean(jack))
    num = float(np.sum((jack_mean - jack) ** 3))
    den = float(np.sum((jack_mean - jack) ** 2))
    a_hat = num / (6.0 * max(den ** 1.5, 1e-12))
    alpha = (100.0 - ci) / 200.0
    z_lo = _norm_ppf(alpha)
    z_hi = _norm_ppf(1 - alpha)
    a1 = _norm_cdf(z0 + (z0 + z_lo) / max(1 - a_hat * (z0 + z_lo), 1e-12))
    a2 = _norm_cdf(z0 + (z0 + z_hi) / max(1 - a_hat * (z0 + z_hi), 1e-12))
    lo = float(np.percentile(boot_means, 100 * max(0, min(a1, 1))))
    hi = float(np.percentile(boot_means, 100 * max(0, min(a2, 1))))
    return theta_hat, lo, hi


def hierarchical_bootstrap_mean_ci(seed_to_values, *, ci=95.0, n_boot=None, seed=0):
    n_boot = n_boot or BOOTSTRAP_N_BCA
    seeds_list = list(seed_to_values.keys())
    if not seeds_list:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_boot):
        chosen_seeds = rng.choice(seeds_list, size=len(seeds_list), replace=True)
        vals = []
        for s in chosen_seeds:
            ep_vals = seed_to_values[s]
            if ep_vals:
                chosen_eps = rng.choice(ep_vals, size=len(ep_vals), replace=True)
                vals.extend(chosen_eps)
        if vals:
            boot_means.append(float(np.nanmean(vals)))
    if not boot_means:
        return float("nan"), float("nan"), float("nan")
    boot_means = np.array(boot_means)
    alpha = (100.0 - ci) / 2.0
    return float(np.mean(boot_means)), float(np.percentile(boot_means, alpha)), float(np.percentile(boot_means, 100 - alpha))


def holm_bonferroni(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    result = {}
    for rank, (name, p) in enumerate(items):
        adj_alpha = alpha / max(m - rank, 1)
        result[name] = {"p": float(p), "rank": rank + 1, "adj_alpha": adj_alpha,
                        "reject": bool(p < adj_alpha)}
    return result


def paired_tests(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.size, y.size)
    if n < 2:
        return {"p_ttest": float("nan"), "p_wilcoxon": float("nan"),
                "mean_diff": float("nan"), "cohen_d": float("nan")}
    x = x[:n]
    y = y[:n]
    diff = x - y
    mean_diff = float(np.mean(diff))
    std_pool = float(np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2))
    cohen_d = mean_diff / max(std_pool, 1e-12)
    p_t = float("nan")
    p_w = float("nan")
    if _HAS_SCIPY:
        try:
            p_t = float(spstats.ttest_rel(x, y).pvalue)
        except Exception:
            pass
        try:
            res = spstats.wilcoxon(diff, alternative="two-sided")
            p_w = float(res.pvalue)
        except Exception:
            pass
    return {"p_ttest": p_t, "p_wilcoxon": p_w,
            "mean_diff": mean_diff, "cohen_d": cohen_d}


def statistical_power_analysis(*, effect_size=0.5, alpha=0.05,
                                n_seeds=3, n_episodes_per_seed=20):
    n_total = n_seeds * n_episodes_per_seed
    se = 1.0 / math.sqrt(max(n_total, 1))
    z_alpha = _norm_ppf(1 - alpha / 2)
    z_beta = effect_size / se - z_alpha
    power_t = _norm_cdf(z_beta)
    power_w = max(0, power_t - 0.05)
    min_d = z_alpha * se * 2
    return {
        "effect_size": effect_size, "alpha": alpha,
        "n_seeds": n_seeds, "n_episodes_per_seed": n_episodes_per_seed,
        "n_total": n_total, "power_ttest": power_t,
        "power_wilcoxon": power_w, "min_detectable_d": min_d,
    }
