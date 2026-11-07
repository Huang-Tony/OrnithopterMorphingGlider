#!/usr/bin/env python3
"""Platform figure (Nature style): 3D renders of the simulated glider + classic block diagram.

Requires the renders from revision/render_glider.py (PyVista) and platform_configs.json.
a  3/4 view of the glider (E387-type sections lofted on the solved cubic-Bezier spars) with labelled parts;
b  morphing configurations (near-plan oblique view) with the moments computed by the aerodynamic model;
c  closed-loop block diagram of one control step.
"""

import json
import os

import numpy as np

import common  # noqa: F401
from common import RUNS_DIR
import figstyle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

RD = os.path.join(RUNS_DIR, "paper_figures", "renders")


def load_cropped(name, pad=8):
    img = plt.imread(os.path.join(RD, f"{name}.png"))
    a = img[..., 3] if img.shape[-1] == 4 else (img[..., :3].min(-1) < 0.99).astype(float)
    ys, xs = np.where(a > 0.02)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0]); x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    return img[y0:y1, x0:x1], (x0, y0, x1, y1, img.shape[1], img.shape[0])


def to_crop(pt, box):
    x0, y0, x1, y1, W, H = box
    return ((pt[0] * W - x0) / (x1 - x0), (pt[1] * H - y0) / (y1 - y0))


def main():
    figstyle.apply()
    ink, ink2, blue, orange, muted = figstyle.INK, figstyle.INK2, figstyle.SLOT["blue"], figstyle.SLOT["orange"], figstyle.MUTED
    cfg = json.load(open(os.path.join(RUNS_DIR, "analysis", "platform_configs.json")))["configs"]
    anchors = json.load(open(os.path.join(RD, "anchors.json")))["hero"]
    fig = plt.figure(figsize=(figstyle.DOUBLE, 4.55))
    # ---------------- a: hero render ----------------
    axa = fig.add_axes([0.0, 0.40, 0.64, 0.58]); axa.axis("off")
    img, box = load_cropped("hero"); axa.imshow(img); h, w = img.shape[:2]
    axa.set_xlim(-0.02 * w, 1.02 * w); axa.set_ylim(1.04 * h, -0.10 * h)
    P = {k: to_crop(v, box) for k, v in anchors.items()}
    def lab(key, text, dx, dy, color=ink, ha="left"):
        x, y = P[key][0] * w, P[key][1] * h
        axa.annotate(text, xy=(x, y), xytext=(x + dx * w, y + dy * h), fontsize=6, color=color, ha=ha, va="center",
                     arrowprops=dict(arrowstyle="-", color=color, lw=0.5, shrinkA=0, shrinkB=2))
    lab("fuselage_nose", "fuselage (avionics, IMU)", -0.02, 0.14, ha="center")
    lab("skin_R", "wing skin\n(E387-type section)", 0.11, -0.10)
    lab("spar_L", "compliant spar\n(cubic Bézier)", -0.13, 0.20, color=blue, ha="center")
    lab("P0", "$P_0$ root (fixed)", 0.03, -0.24, color=blue, ha="center")
    lab("R_P1", "$P_1, P_2$ solved by\nenergy minimisation", 0.12, -0.17, color=blue)
    lab("R_P3", "$P_3$ commanded tip\n$(\\Delta x, \\Delta y, \\Delta z)$", -0.20, 0.04, color=orange, ha="center")
    axa.text(0.0, 1.0, "a", transform=axa.transAxes, fontsize=8, fontweight="bold", va="top")
    # ---------------- b: configurations ----------------
    items = [("cfg_neutral_front", "neutral, front view: gull-arch rest shape", None),
             ("cfg_diff_sweep_top", "differential sweep ±0.30 m, top view (nose up) → yaw", ("z", cfg["diff_sweep"]["dM"][2])),
             ("cfg_diff_dihedral_front", "differential dihedral ±0.12 m, front view → roll", ("x", cfg["diff_dihedral"]["dM"][0]))]
    tops = [0.955, 0.80, 0.555]; hts = [0.10, 0.205, 0.10]
    for i, (nm, title, mom) in enumerate(items):
        ax = fig.add_axes([0.668, tops[i] - hts[i] - 0.025, 0.322, hts[i]]); ax.axis("off")
        im, _ = load_cropped(nm); ax.imshow(im)
        txt = title + (f"\n$\\Delta M_{mom[0]}$ = {mom[1]:+.3f} N·m (model)" if mom else "")
        fig.text(0.665, tops[i], txt, fontsize=5.7, va="top", color=ink, linespacing=1.3)
    fig.text(0.648, 0.98, "b", fontsize=8, fontweight="bold", va="top")
    # ---------------- c: block diagram ----------------
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.37]); ax.axis("off"); ax.set_xlim(0, 183); ax.set_ylim(0, 62)
    ax.text(1, 61, "c", fontsize=8, fontweight="bold", va="top")

    def block(x, y, w_, h_, name, sub=None):
        ax.add_patch(Rectangle((x, y), w_, h_, fc="white", ec=ink, lw=0.6))
        ax.text(x + w_ / 2, y + h_ / 2 + (2.3 if sub else 0), name, ha="center", va="center", fontsize=6.2)
        if sub: ax.text(x + w_ / 2, y + h_ / 2 - 3.2, sub, ha="center", va="center", fontsize=5.3, color=ink2)

    def sig(p, q, label=None, pos=0.5, off=(0, 1.6), ha="center"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, color=ink, lw=0.6, shrinkA=0, shrinkB=0))
        if label:
            ax.text(p[0] + (q[0] - p[0]) * pos + off[0], p[1] + (q[1] - p[1]) * pos + off[1], label, ha=ha, va="bottom",
                    fontsize=6.2, style="italic")

    def line(pts):
        pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], color=ink, lw=0.6, solid_capstyle="butt")

    y = 36; hb = 14
    ax.add_patch(Rectangle((41, 5), 140, 51, fc="none", ec=muted, lw=0.6, ls=(0, (4, 2))))
    ax.text(43, 54.5, "simulated plant (control step 40 ms; rigid-body integration 4 × 10 ms)", ha="left", va="top",
            fontsize=5.6, color=muted)
    block(8, y, 22, hb, "policy $\\pi_\\theta$", "SAC actor (MLP)")
    block(45, y, 25, hb, "actuators", "first-order lag, $\\tau$")
    block(78, y, 25, hb, "spar solver", "$P_1, P_2 = \\arg\\min J$")
    block(111, y, 29, hb, "aerodynamics", "strip theory, 2 × 10 strips")
    block(148, y, 29, hb, "rigid-body dynamics", "6 DOF, quaternion")
    block(148, 9, 29, hb, "sensors", "IMU, airspeed, altitude")
    block(78, 9, 50, hb, "observation", "attitude, rates, airspeed, altitude,\nreference, wing geometry")
    sig((30, y + hb / 2), (45, y + hb / 2), "$a_t$")
    sig((70, y + hb / 2), (78, y + hb / 2), "$P_3$")
    sig((103, y + hb / 2), (111, y + hb / 2), "$P_{1,2}$")
    sig((140, y + hb / 2), (148, y + hb / 2), "$F, M$")
    sig((162.5, y), (162.5, 9 + hb), "$x$", pos=0.5, off=(2.5, -1.2), ha="left")
    sig((148, 9 + hb / 2), (128, 9 + hb / 2), "$y$", off=(0, 0.8))
    line([(78, 9 + hb / 2), (19, 9 + hb / 2)]); sig((19, 9 + hb / 2), (19, y), "$\\tilde s_t$", pos=0.45, off=(-2.5, -1.2), ha="right")
    sig((0.5, y + hb - 3.5), (8, y + hb - 3.5), "$r_{ref}$", off=(-0.2, 0.6))
    sig((125.5, 61.5), (125.5, y + hb), None); ax.text(127, 61.5, "gusts, wind, randomised physics", fontsize=5.6,
                                                        color=ink2, va="top")
    figstyle.save(fig, os.path.join(RUNS_DIR, "paper_figures", "fig_platform"))
    print("saved")


if __name__ == "__main__":
    main()
