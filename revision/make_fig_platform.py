#!/usr/bin/env python3
"""Platform figure (Nature style), revision R1.

a  three-quarter render of the simulated glider (revision/render_glider.py) with the spar control points of one wing,
   the commanded tip degrees of freedom, and the body axes;
b  vector small multiples of four morphing modes, each drawn in the view in which its moment is an in-plane
   rotation (neutral shape ghosted, morphed shape coloured, tip motion arrows, moment glyph with the value computed
   by the aerodynamic model about the centre of mass; signs follow the computed moments);
c  closed-loop block diagram of one control step, colour-coded by role.
Inputs: analysis/platform_configs.json (make_platform_geometry.py), paper_figures/renders/{hero.png, anchors.json}.
"""

import json
import os

import numpy as np

import common  # noqa: F401
from common import RUNS_DIR
import figstyle
FS = 1.35                      # revision R1: lettering scale so that the printed size at the text width is >= ~5.5 pt
W_FIG = 183 * figstyle.MM      # layout designed at 183 mm; included at the 131 mm text width
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Ellipse, Arc
from glider_mesh import wing_grid, bezier, fuselage_grid

RD = os.path.join(RUNS_DIR, "paper_figures", "renders")
INK, INK2, MUTED = figstyle.INK, figstyle.INK2, figstyle.MUTED
BLUE, ORANGE = figstyle.SLOT["blue"], figstyle.SLOT["orange"]
GHOST_F, GHOST_E = "#eceef1", "#b9c0c8"
WING_F, WING_E = "#d6e5f6", BLUE
FUS_F = "#c9ced4"
CTRL_F, PLANT_F, ENV = "#e3eefa", "#f1f2f4", "#8a5a2b"


def load_cropped(name, pad=10):
    img = plt.imread(os.path.join(RD, f"{name}.png"))
    a = img[..., 3] if img.shape[-1] == 4 else (img[..., :3].min(-1) < 0.99).astype(float)
    ys, xs = np.where(a > 0.02)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    return img[y0:y1, x0:x1], (x0, y0, x1, y1, img.shape[1], img.shape[0])


def to_crop(pt, box):
    x0, y0, x1, y1, W, H = box
    return (pt[0] * W - x0, pt[1] * H - y0)


# ------------------------------------------------------------------ projections (body frame: x fwd, y to 'R', z up)
VIEWS = {"top": lambda P: np.stack([-P[..., 1], P[..., 0]], -1),      # looking down, nose up (+y wing on the left)
         "front": lambda P: np.stack([P[..., 1], P[..., 2]], -1),     # looking aft from ahead (+y wing on the right)
         "side": lambda P: np.stack([P[..., 0], P[..., 2]], -1)}      # from the -y side, nose to the right
# sign of a positive body moment as seen in each view (+1: counter-clockwise)
CCW = {"top": {"z": +1}, "front": {"x": +1}, "side": {"y": -1}}


def planform(P):
    X = wing_grid(np.asarray(P), n_span=60, n_af=40); n = X.shape[1]
    return np.vstack([X[:, n // 2, :], X[::-1, 0, :]])            # leading edge root->tip, trailing edge tip->root


def draw_view(ax, cfg_neu, cfg, view, *, mom_axis, mom_val, title, subtitle, show_axes):
    proj = VIEWS[view]
    for c, fc, ec, lw, z in ((cfg_neu, GHOST_F, GHOST_E, 0.5, 1), (cfg, WING_F, WING_E, 0.6, 3)):
        for side in ("R", "L"):
            P = np.asarray(c[side])
            if view == "top":
                ax.add_patch(Polygon(proj(planform(P)), closed=True, fc=fc, ec=ec, lw=lw, zorder=z))
                ax.plot(*proj(bezier(P, np.linspace(0, 1, 60))).T, color=ec, lw=lw * 1.4, zorder=z + 0.5)
            elif view == "front":
                ax.plot(*proj(bezier(P, np.linspace(0, 1, 80))).T, color=ec, lw=2.2 if z > 1 else 1.8,
                        solid_capstyle="round", zorder=z, alpha=1.0 if z > 1 else 0.9)
            else:   # side: spar seen from the side (both wings coincide for symmetric inputs)
                ax.plot(*proj(bezier(P, np.linspace(0, 1, 80))).T, color=ec, lw=2.2 if z > 1 else 1.8,
                        solid_capstyle="round", zorder=z + 2 if z > 1 else z)
    F = fuselage_grid(n_x=50, n_t=24)
    if view == "front":
        ax.add_patch(Ellipse((0, -0.018), 0.06, 0.051, fc=FUS_F, ec=INK2, lw=0.4, zorder=5))
    else:
        top = proj(F[:, F.shape[1] // 4, :]); bot = proj(F[:, 3 * F.shape[1] // 4, :])
        if view == "top":
            top = proj(F[:, 0, :]); bot = proj(F[:, F.shape[1] // 2, :])
        ax.add_patch(Polygon(np.vstack([top, bot[::-1]]), closed=True, fc=FUS_F, ec=INK2, lw=0.4, zorder=4))
    for side in ("R", "L"):                                       # tip motion
        a, b = proj(np.asarray(cfg_neu[side][3])), proj(np.asarray(cfg[side][3]))
        if np.linalg.norm(b - a) > 1e-3:
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, color=ORANGE, lw=0.8, zorder=8,
                                         shrinkA=0, shrinkB=0))
            ax.plot(*a, "o", ms=1.8, mfc="white", mec=ORANGE, mew=0.6, zorder=8)
    # moment glyph about the centre of mass
    s = np.sign(mom_val) * CCW[view][mom_axis]
    r = 0.17; cen = (0.0, 0.0) if view != "front" else (0.0, -0.018)
    t1, t2 = (200, 340) if view != "side" else (20, 160)
    th = np.radians(np.linspace(t1, t2, 40)); xy = np.c_[cen[0] + r * np.cos(th), cen[1] + r * 0.62 * np.sin(th)]
    if s < 0: xy = xy[::-1]
    ax.plot(xy[:-2, 0], xy[:-2, 1], color=INK, lw=0.7, zorder=9)
    ax.add_patch(FancyArrowPatch(xy[-4], xy[-1], arrowstyle="-|>", mutation_scale=6, color=INK, lw=0.7, zorder=9,
                                 shrinkA=0, shrinkB=0))
    if show_axes:                                                  # small orientation triad
        o = np.array(show_axes["o"]); L = 0.16
        for (d, lbl) in show_axes["dirs"]:
            e = o + L * np.array(d)
            ax.add_patch(FancyArrowPatch(o, e, arrowstyle="-|>", mutation_scale=4, color=MUTED, lw=0.5,
                                         shrinkA=0, shrinkB=0))
            ax.text(*(o + (L + 0.06) * np.array(d)), lbl, fontsize=5.2*FS, color=MUTED, ha="center", va="center")
    ax.set_aspect("equal"); ax.axis("off")


def main():
    figstyle.apply()
    D = json.load(open(os.path.join(RUNS_DIR, "analysis", "platform_configs.json")))["configs"]
    anchors = json.load(open(os.path.join(RD, "anchors.json")))["hero"]
    fig = plt.figure(figsize=(W_FIG, 4.45))

    # ---------------- a: hero render ----------------
    axa = fig.add_axes([0.0, 0.45, 0.575, 0.53]); axa.axis("off")
    img, box = load_cropped("hero"); axa.imshow(img); h, w = img.shape[:2]
    axa.set_xlim(-0.02 * w, 1.04 * w); axa.set_ylim(1.00 * h, -0.03 * h)
    P = {k: to_crop(v, box) for k, v in anchors.items()}

    def lab(keys, text, xy_text, color=INK, ha="center", va="center", fs=6.0*FS):
        keys = [keys] if isinstance(keys, str) else keys
        rel = {"top": 1.0, "bottom": 0.0, "center": 0.5}
        for i, k in enumerate(keys):          # leader lines start at the edge of the text box facing the target
            ry = 0.0 if P[k][1] / h > xy_text[1] else 1.0           # image y grows downward
            axa.annotate(text, xy=P[k], xytext=(xy_text[0] * w, xy_text[1] * h), fontsize=fs, ha=ha, va=va,
                         color=color if i == 0 else (0, 0, 0, 0), linespacing=1.15,
                         arrowprops=dict(arrowstyle="-", color=color, lw=0.45, shrinkA=1.5, shrinkB=1.5,
                                         relpos=(0.5, 1.0 - ry)))
    lab("skin_L", "wing skin\n(E387-type section)", (0.53, 0.10))
    lab(["R_P1", "R_P2"], "$P_1$, $P_2$ solved by\nenergy minimization", (0.78, 0.20), color=BLUE)
    lab("spar_R", "compliant spar\n(cubic Bézier)", (0.97, 0.44), color=BLUE)
    lab("P0", "$P_0$ root (fixed)", (0.58, 0.82), color=BLUE)
    lab("fuselage_nose", "fuselage (avionics, IMU)", (0.27, 0.85))
    x3, y3 = P["R_P3"]
    axa.text(x3 - 0.17 * w, y3 + 0.02 * h, "$P_3$ commanded tip", fontsize=6.0*FS, color=ORANGE, ha="center", va="center",
             linespacing=1.1, bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85))
    for k, t in (("tip_dx", "$\\Delta x$"), ("tip_dy", "$\\Delta y$"), ("tip_dz", "$\\Delta z$")):
        x, y = P[k]; axa.text(x, y, t, fontsize=5.6*FS, color=ORANGE, ha="center", va="center",
                              bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.8))
    for k, t in (("ax_x", "$x_b$"), ("ax_y", "$y_b$"), ("ax_z", "$z_b$")):
        x, y = P[k]; axa.text(x, y, t, fontsize=5.6*FS, color=INK2, ha="center", va="center")
    fig.text(0.005, 0.985, "a", fontsize=8*FS, fontweight="bold", va="top")

    # ---------------- b: morphing modes ----------------
    fig.text(0.578, 0.985, "b", fontsize=8*FS, fontweight="bold", va="top")
    neu = D["neutral"]
    modes = [("diff_sweep", "top", "z", "Differential sweep", "top view, nose up",
              {"o": (1.22, -0.62), "dirs": [((0, 1), "$x_b$"), ((-1, 0), "$y_b$")]}),
             ("sym_sweep", "side", "y", "Symmetric sweep", "side view, nose right",
              {"o": (0.26, -0.13), "dirs": [((1, 0), "$x_b$"), ((0, 1), "$z_b$")]}),
             ("diff_dihedral", "front", "x", "Differential dihedral", "front view",
              {"o": (-1.28, -0.20), "dirs": [((1, 0), "$y_b$"), ((0, 1), "$z_b$")]}),
             ("diff_span", "front", "x", "Differential span", "front view", None)]
    lims = {"top": ((-1.32, 1.32), (-0.68, 0.44)), "side": ((-0.48, 0.48), (-0.16, 0.30)),
            "front": ((-1.36, 1.36), (-0.24, 0.32))}
    names = {"x": "roll", "y": "pitch", "z": "yaw"}
    cells = [(0.598, 0.715), (0.805, 0.715), (0.598, 0.425), (0.805, 0.425)]
    for (name, view, axn, title, sub, tri), (x0, y0) in zip(modes, cells):
        tight = view == "front"                     # wide, flat views: shorter cell so labels stay close
        ax = fig.add_axes([x0, y0 + (0.075 if tight else 0.035), 0.19, 0.11 if tight else 0.165])
        mv = D[name]["dM"]["xyz".index(axn)]
        draw_view(ax, neu, D[name], view, mom_axis=axn, mom_val=mv, title=title, subtitle=sub, show_axes=tri)
        (xa, xb), (ya, yb) = lims[view]; ax.set_xlim(xa, xb); ax.set_ylim(ya, yb)
        fig.text(x0, y0 + (0.228 if tight else 0.262), title, fontsize=6.2*FS, fontweight="bold", color=INK, va="top")
        fig.text(x0, y0 + (0.202 if tight else 0.236), sub, fontsize=5.4*FS, color=MUTED, va="top")
        fig.text(x0 + 0.095, y0 + (0.080 if tight else 0.030), f"$\\Delta M_{axn}$ = {mv:+.2f} N m ({names[axn]})", fontsize=5.6*FS,
                 color=INK, ha="center", va="top")

    # ---------------- c: block diagram ----------------
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.445]); ax.axis("off"); ax.set_xlim(0, 183); ax.set_ylim(0, 68)
    ax.text(1, 67, "c", fontsize=8*FS, fontweight="bold", va="top")

    def block(x, y, w_, h_, name, sub=None, fc=PLANT_F, ec=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w_, h_, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=0.6))
        ax.text(x + w_ / 2, y + h_ / 2 + (2.4 if sub else 0), name, ha="center", va="center", fontsize=6.2*FS, color=INK)
        if sub: ax.text(x + w_ / 2, y + h_ / 2 - 2.9, sub, ha="center", va="center", fontsize=5.2*FS, color=INK2)

    def sig(pts, label=None, at=0.5, off=(0, 1.3), ha="center", va="bottom", color=INK):
        pts = np.asarray(pts, float)
        if len(pts) > 2: ax.plot(pts[:-1, 0], pts[:-1, 1], color=color, lw=0.6, solid_capstyle="butt")
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=6, color=color, lw=0.6,
                                     shrinkA=0, shrinkB=0))
        if label:
            p, q = pts[0], pts[1]
            ax.text(p[0] + (q[0] - p[0]) * at + off[0], p[1] + (q[1] - p[1]) * at + off[1], label, ha=ha, va=va,
                    fontsize=5.6*FS, color=color)

    yb, hb = 38, 13
    ax.add_patch(FancyBboxPatch((40, 4), 141, 56, boxstyle="round,pad=0,rounding_size=2.0", fc="none", ec=MUTED,
                                lw=0.6, ls=(0, (3, 2))))
    ax.text(42.5, 30.0, "Simulated plant  ($\\Delta t$ = 40 ms control step, 4 × 10 ms integration substeps)",
            ha="left", va="center", fontsize=5.5*FS, color=MUTED)
    block(7, yb, 24, hb, "Policy $\\pi_\\theta$", "SAC actor (MLP)", fc=CTRL_F, ec=BLUE)
    block(46, yb, 23, hb, "Actuators", "lag $\\tau$ = 70 ms")
    block(78, yb, 23, hb, "Spar solver", "$P_1, P_2 = \\arg\\min J$")
    block(110, yb, 28, hb, "Aerodynamics", "strip theory, $c_l$, $c_d$, $c_{m0}$")
    block(147, yb, 30, hb, "Rigid-body dynamics", "6 DOF, quaternion")
    block(147, 9, 30, hb, "Sensors", "IMU, airspeed, altitude")
    block(78, 9, 60, hb, "Observation", "attitude, rates, air data, reference, wing shape (41-D)")
    m = yb + hb / 2
    sig([(31, m), (46, m)], "$\\mathbf{a}_t$", at=0.3)
    sig([(69, m), (78, m)], "$P_3$")
    sig([(101, m), (110, m)], "$P_1,P_2$")
    sig([(138, m), (147, m)], "$\\mathbf{F}_b,\\mathbf{M}_b$")
    sig([(162, yb), (162, 9 + hb)], "state", at=0.5, off=(1.5, 0), ha="left", va="center")
    sig([(147, 9 + hb / 2), (138, 9 + hb / 2)], "$\\mathbf{y}$", off=(0, 0.8))
    sig([(78, 9 + hb / 2), (19, 9 + hb / 2), (19, yb)])
    ax.text(20.5, 27, "$\\tilde{\\mathbf{s}}_t$", fontsize=6.0*FS, ha="left", va="center")
    sig([(0.5, yb + hb - 3.5), (7, yb + hb - 3.5)], "$r_{\\mathrm{ref}}$", off=(-0.5, 0.6))
    sig([(124, 66.5), (124, yb + hb)], None, color=ENV)
    ax.text(125.5, 66.5, "gusts, wind, randomized parameters", fontsize=5.5*FS, color=ENV, va="top")
    figstyle.save(fig, os.path.join(RUNS_DIR, "paper_figures", "fig_platform"))
    print("saved")


if __name__ == "__main__":
    main()
