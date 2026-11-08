#!/usr/bin/env python3
"""Plan A validation: compare the aerodynamic proxy's sectional lift/drag law with public wind-tunnel data
(Eppler 387, NASA Langley LTPT, NASA TM-4062, Re = 1e5 and 2e5; the glider operates at Re ~ 1.5e5).

The proxy has no induced-downwash model, so its effective sectional slope s*c_la (lift_scale x c_l_alpha)
plays the role of the finite-wing lift slope; it is compared with the Helmbold lifting-surface estimate
built from the measured 2-D slope for the glider's aspect ratio (AR = 2b/c = 13.3).
Outputs: revision_runs/analysis/aero_validation_e387.json, paper figure fig_aero_validation.{pdf,png}.
"""

import csv
import math
import os

import numpy as np

import common  # noqa: F401
from common import RUNS_DIR, REPO, dump_json
from morphing_glider.config import WING_CHORD, L_FIXED
from morphing_glider.physics.domain_randomizer import NOMINAL_PHYS, DomainRandomizer
import figstyle
import matplotlib.pyplot as plt

AR = 2 * L_FIXED / WING_CHORD


def load_e387():
    rows = [r for r in csv.reader(l for l in open(os.path.join(REPO, "revision", "data", "e387_ltpt_tm4062.csv"))
                                   if not l.startswith("#"))]
    hdr = rows[0]; data = {}
    for r in rows[1:]:
        d = dict(zip(hdr, r)); Re = int(d["reynolds"])
        data.setdefault(Re, []).append((float(d["alpha_deg"]), float(d["cl"]), float(d["cd"]) if d["cd"] else np.nan))
    return {Re: np.array(sorted(v)) for Re, v in data.items()}


def helmbold(a0):
    k = a0 / (math.pi * AR)
    return a0 / (math.sqrt(1 + k * k) + k)


def proxy_cl(alpha_rad, ph):
    a = np.clip(alpha_rad + ph["alpha0"], -ph["alpha_clip"], ph["alpha_clip"])
    if ph.get("cl_sat_n") is None:
        return ph["lift_scale"] * ph["cl_max"] * np.tanh(ph["cl_alpha"] * a / ph["cl_max"])
    x = ph["cl_alpha"] * a; n = ph["cl_sat_n"]
    cl = x / (1.0 + np.abs(x / ph["cl_max"]) ** n) ** (1.0 / n)
    z = np.clip((np.abs(a) - ph["alpha_stall"]) / 0.03, 0.0, 1.0)
    cl = cl * (1.0 - (1.0 - ph.get("post_stall_cl_factor", 1.0)) * z * z * (3 - 2 * z))
    return ph["lift_scale"] * cl


def v1_phys():
    import json
    v1 = json.load(open(os.path.join(REPO, "revision", "data", "aero_calibration_e387.json")))["v1_values"]
    ph = dict(NOMINAL_PHYS); ph.update(v1); ph.pop("cl_sat_n", None); ph["post_stall_cl_factor"] = 1.0
    return ph


def proxy_cd(cl_eff, alpha_rad, ph):
    a = np.clip(alpha_rad + ph["alpha0"], -ph["alpha_clip"], ph["alpha_clip"])
    cl = cl_eff / ph["lift_scale"]
    stall = np.maximum(0.0, np.abs(a) - ph["alpha_stall"]) / ph["alpha_clip"]
    return ph["cd0"] + ph["k_induced"] * cl ** 2 + ph["cd_stall"] * stall ** 2


def main():
    figstyle.apply()
    E = load_e387(); ph = dict(NOMINAL_PHYS); out = {"AR": AR, "source": "NASA TM-4062 Table B1"}
    for Re, d in E.items():
        a, cl, cd = d[:, 0], d[:, 1], d[:, 2]
        lin = (a >= -3.0) & (a <= 6.1)
        p = np.polyfit(np.radians(a[lin]), cl[lin], 1); a0 = float(p[0]); aL0 = float(-p[1] / p[0])
        i_max = int(np.argmax(cl)); post = np.where((a > a[i_max]) & (cl < 0.85 * cl[i_max]))[0]
        out[f"Re{Re}"] = {"a0_2D_per_rad": a0, "alpha_L0_deg": math.degrees(aL0), "cl_max": float(cl[i_max]),
                          "alpha_clmax_deg": float(a[i_max]),
                          "alpha_stall_drop_deg": float(a[post[0]]) if post.size else None,
                          "cd_min": float(np.nanmin(cd)), "a_3D_helmbold_per_rad": helmbold(a0)}
    out["v1_proxy"] = {k: v for k, v in v1_phys().items() if k in ("cl_alpha", "lift_scale", "cl_max", "cd0", "k_induced", "alpha0", "alpha_stall")}
    s_cla = ph["lift_scale"] * ph["cl_alpha"]
    out["proxy"] = {"cl_alpha_local": ph["cl_alpha"], "lift_scale": ph["lift_scale"], "effective_slope_per_rad": s_cla,
                    "effective_cl_max": ph["lift_scale"] * ph["cl_max"], "alpha0_deg": math.degrees(ph["alpha0"]),
                    "alpha_stall_deg": math.degrees(ph["alpha_stall"]), "cd0": ph["cd0"], "k_induced": ph["k_induced"]}
    for Re in E:
        r = out[f"Re{Re}"]
        out[f"Re{Re}"]["proxy_slope_rel_err"] = s_cla / r["a_3D_helmbold_per_rad"] - 1.0
        out[f"Re{Re}"]["proxy_clmax_rel_err"] = out["proxy"]["effective_cl_max"] / r["cl_max"] - 1.0
        # both angles measured from zero lift (the proxy's alpha_stall acts on alpha + alpha0)
        out[f"Re{Re}"]["proxy_stall_minus_meas_deg"] = out["proxy"]["alpha_stall_deg"] - ((r["alpha_stall_drop_deg"] or np.nan) - r["alpha_L0_deg"])
    # domain-randomisation envelope of the proxy law (rand_scale = 1)
    rng = np.random.default_rng(0); dr = DomainRandomizer(enabled=True, scale=1.0)
    draws = [dr.sample(rng) for _ in range(2000)]
    slopes = np.array([d_["lift_scale"] * d_["cl_alpha"] for d_ in draws])
    clmaxs = np.array([d_["lift_scale"] * d_["cl_max"] for d_ in draws])
    out["randomisation_envelope"] = {"effective_slope_p5_p95": np.percentile(slopes, [5, 95]).tolist(),
                                     "effective_slope_min_max": [float(slopes.min()), float(slopes.max())],
                                     "effective_clmax_p5_p95": np.percentile(clmaxs, [5, 95]).tolist()}
    for Re in E:
        d = E[Re]; ok = np.isfinite(d[:, 2])
        for clq in (0.2, 0.4, 0.6):
            cd_meas = float(np.interp(clq, d[ok, 1], d[ok, 2]) + clq ** 2 / (math.pi * 0.95 * AR))
            cd_prox = float(ph["cd0"] + ph["k_induced"] * (clq / ph["lift_scale"]) ** 2)
            out[f"Re{Re}"][f"drag_ratio_proxy_over_meas3D_at_cl{clq}"] = cd_prox / cd_meas
    out["proxy_oswald_equiv"] = 1.0 / (math.pi * AR * ph["k_induced"] / ph["lift_scale"] ** 2)
    out["helmbold_slopes_inside_envelope"] = all(slopes.min() <= out[f"Re{Re}"]["a_3D_helmbold_per_rad"] <= slopes.max() for Re in E)
    dump_json(out, os.path.join(RUNS_DIR, "analysis", "aero_validation_e387.json"))

    # ---- figure: (a) lift vs angle relative to zero-lift angle, (b) drag polar ----
    fig, axs = plt.subplots(1, 2, figsize=(figstyle.DOUBLE, 2.35))
    ax = axs[0]; aa = np.radians(np.linspace(-6, 20, 300))
    # proxy, plotted against angle from its own zero-lift angle (alpha + alpha0)
    curves = np.array([proxy_cl(aa - ph["alpha0"], d_) for d_ in draws[:400]])
    x_proxy = np.degrees(aa)
    ax.fill_between(x_proxy, np.percentile(curves, 5, 0), np.percentile(curves, 95, 0), color=figstyle.LIGHT, alpha=0.55,
                    lw=0, label="v2, randomized 5–95%")
    ax.plot(x_proxy, proxy_cl(aa - ph["alpha0"], ph), color=figstyle.INK, lw=1.0, label="Proxy v2 (calibrated)")
    p1 = v1_phys(); ax.plot(x_proxy, proxy_cl(aa - p1["alpha0"], p1), color=figstyle.MUTED, lw=0.9, ls="--",
                            label="Proxy v1 (submitted)")
    for Re, col in ((100000, figstyle.SLOT["blue"]), (200000, figstyle.SLOT["orange"])):
        d = E[Re]; aL0 = out[f"Re{Re}"]["alpha_L0_deg"]
        ax.plot(d[:, 0] - aL0, d[:, 1], "o", ms=2.6, mfc="white", mec=col, mew=0.7,
                label=f"E387, Re = {Re/1e5:.0f}×10$^5$ (2-D)")
        a3 = out[f"Re{Re}"]["a_3D_helmbold_per_rad"]
        xs = np.linspace(0, 9, 10); ax.plot(xs, a3 * np.radians(xs), "--", color=col, lw=0.7)
    ax.set_xlabel(r"$\alpha - \alpha_{L0}$ (deg)"); ax.set_ylabel(r"$c_l$"); ax.set_xlim(-4, 20); ax.set_ylim(-0.3, 1.45)
    ax.legend(loc="lower right", fontsize=5.8); figstyle.panel_label(ax, "a")
    ax.text(-3.4, 1.36, "colored dashes: finite-wing\nslope from 2-D data\n(Helmbold, AR = 13.3)", fontsize=5.5,
            color=figstyle.INK2, va="top")
    ax = axs[1]
    cl_p = np.linspace(0.0, 1.15, 100)
    cd_curves = np.array([d_["cd0"] + d_["k_induced"] * (cl_p / d_["lift_scale"]) ** 2 for d_ in draws[:400]])
    ax.fill_between(cl_p, np.percentile(cd_curves, 5, 0), np.percentile(cd_curves, 95, 0), color=figstyle.LIGHT, alpha=0.55, lw=0)
    ax.plot(cl_p, ph["cd0"] + ph["k_induced"] * (cl_p / ph["lift_scale"]) ** 2, color=figstyle.INK, lw=1.0)
    p1 = v1_phys(); ax.plot(cl_p, p1["cd0"] + p1["k_induced"] * (cl_p / p1["lift_scale"]) ** 2, color=figstyle.MUTED, lw=0.9, ls="--")
    e_osw = 0.95
    for Re, col in ((100000, figstyle.SLOT["blue"]), (200000, figstyle.SLOT["orange"])):
        d = E[Re]; ok = np.isfinite(d[:, 2])
        cd3 = d[ok, 2] + d[ok, 1] ** 2 / (math.pi * e_osw * AR)      # profile + ideal induced drag
        ax.plot(d[ok, 1], cd3, "o", ms=2.6, mfc="white", mec=col, mew=0.7)
    ax.text(0.64, 0.002, "E387: measured profile\ndrag + ideal induced drag\n$c_l^2/(\\pi e\\,AR)$, e = 0.95", fontsize=5.5,
            color=figstyle.INK2, va="bottom")
    ax.set_xlabel(r"$c_l$"); ax.set_ylabel(r"$c_d$ (section, incl. induced)"); ax.set_ylim(0, 0.08); figstyle.panel_label(ax, "b")
    fig.tight_layout(w_pad=2.0); figstyle.save(fig, os.path.join(RUNS_DIR, "paper_figures", "fig_aero_validation"))
    for Re in E:
        r = out[f"Re{Re}"]
        print(f"Re={Re}: a0={r['a0_2D_per_rad']:.2f}/rad aL0={r['alpha_L0_deg']:.2f} deg clmax={r['cl_max']:.3f} @ {r['alpha_clmax_deg']:.1f} "
              f"stall-drop@{r['alpha_stall_drop_deg']} a3D={r['a_3D_helmbold_per_rad']:.2f} | proxy slope err {100*r['proxy_slope_rel_err']:+.1f}% "
              f"clmax err {100*r['proxy_clmax_rel_err']:+.1f}% stall diff {r['proxy_stall_minus_meas_deg']:+.1f} deg")
    print("proxy:", {k: round(v, 3) for k, v in out["proxy"].items()})
    print("drag ratios:", {k: {kk: round(vv, 2) for kk, vv in v.items() if kk.startswith("drag_ratio")} for k, v in out.items() if k.startswith("Re")},
          "proxy Oswald-equivalent e =", round(out["proxy_oswald_equiv"], 3))
    print("randomisation envelope:", out["randomisation_envelope"], "Helmbold inside:", out["helmbold_slopes_inside_envelope"])


if __name__ == "__main__":
    main()
