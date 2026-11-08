#!/usr/bin/env python3
"""Render the supplementary videos (revision R1, Reviewer 2). Runs in the render environment (pyvista):
    <render_venv>/bin/python revision/make_videos.py [--only S1 S2 ...] [--fps 25] [--quick]

Input: revision_runs/videos/data/*.npz (from video_episodes.py, all on the same evaluation episode).
  S1  trained SAC policy (teacher): chase-camera 3-D view + yaw-rate tracking, attitude, tip commands
  S2  SAC teacher vs best tuned classical controller, side by side, same episode
  S3  training progress: SAC seed 7 checkpoints flying the same episode
  S4  SAC teacher vs distilled KAN student, side by side
  S5  SAC (1M steps) vs SAC-S (smoothness fine-tuned, smooth_study.py), side by side; S1 and S2 show SAC-S when available
Each frame shows the simulated state at that control step (40 ms); videos play in real time at 25 fps.
Output: revision_runs/videos/S<k>_<name>.mp4 (H.264, yuv420p).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from glider_mesh import wing_grid, fuselage_grid, bezier, euler_to_R  # noqa: E402

RUNS = os.environ.get("REVISION_RUNS_DIR", os.path.join(os.path.dirname(HERE), "revision_runs"))
DATA = os.path.join(RUNS, "videos", "data"); OUT = os.environ.get("VIDEO_OUT", os.path.join(RUNS, "videos"))
DT = 0.04
SKIN = "#a9bccf"; SKIN_EDGE = "#5d6b79"; SPAR = "#1f5fae"; FUS = "#b9bec5"
C_REF = "#4d4d4d"; C_A = "#2a78d6"; C_B = "#d9542b"; C_C = "#1b9e77"
FUS_GRID = None


def load(tag):
    d = np.load(os.path.join(DATA, f"{tag}.npz")); meta = json.load(open(os.path.join(DATA, f"{tag}.json")))
    ep = {k: d[k][0] for k in d.files}
    n = int(np.sum(np.isfinite(ep["yaw_rate"])))
    ep = {k: v[:n] for k, v in ep.items()}
    ep["rms_h"] = float(meta["episodes"][0]["rms_yaw_horizon"]); ep["failed"] = bool(meta["episodes"][0]["failure"])
    return ep


def body_cfg(ep, t):
    cfg = {}
    for side, key in (("R", "R"), ("L", "L")):
        p3 = ep[f"p3_{key}"][t]; p1 = ep[f"spar_{key}_p1"][t]; p2 = ep[f"spar_{key}_p2"][t]
        cfg[side] = np.array([[0.0, 0.0, 0.0], p1, p2, p3])
    return cfg


def add_glider(p, ep, t, pv):
    """Glider at step t in world coordinates (position re-centred on the start point)."""
    global FUS_GRID
    R = euler_to_R(ep["roll"][t], ep["pitch"][t], ep["yaw"][t]); x0 = ep["pos"][t] - ep["pos"][0]
    tf = lambda X: (X.reshape(-1, 3) @ R.T + x0).reshape(X.shape)
    def surf(X):
        g = pv.StructuredGrid(X[..., 0], X[..., 1], X[..., 2])
        return g.extract_surface().triangulate()
    cfg = body_cfg(ep, t)
    for side in ("R", "L"):
        X = tf(wing_grid(cfg[side], n_span=41, n_af=32))
        p.add_mesh(surf(X), color=SKIN, opacity=0.95, smooth_shading=True, specular=0.3, ambient=0.25, diffuse=0.8)
        p.add_mesh(pv.Spline(tf(bezier(cfg[side], np.linspace(0, 1, 60))), 120).tube(radius=0.008), color=SPAR,
                   smooth_shading=True)
        tip = tf(cfg[side][3][None])[0]
        p.add_mesh(pv.Sphere(radius=0.022, center=tip), color=C_B, smooth_shading=True)
    if FUS_GRID is None: FUS_GRID = fuselage_grid(n_x=40, n_t=24)
    p.add_mesh(surf(tf(FUS_GRID)), color=FUS, smooth_shading=True, specular=0.3, ambient=0.3)
    return R, x0


ARC_T = 2.0          # [s] the yaw arcs show the heading change over this horizon at the given yaw rate


def add_yaw_arc(p, pv, x0, psi, rate, radius, color, tube_r=0.016, opacity=1.0):
    """Horizontal arc around the glider from the current heading psi to psi + rate * ARC_T, with an arrowhead.
    Positive yaw rate (z up) turns the nose to the left (counter-clockwise seen from above). A rate below
    0.02 rad/s is drawn as a short straight arrow ahead of the nose ('hold heading')."""
    sweep = float(rate) * ARC_T
    if abs(sweep) < 0.04:
        a = x0 + (radius - 0.25) * np.array([np.cos(psi), np.sin(psi), 0.0])
        b = x0 + (radius + 0.05) * np.array([np.cos(psi), np.sin(psi), 0.0])
        p.add_mesh(pv.Line(a, b).tube(radius=tube_r), color=color, opacity=opacity, smooth_shading=True)
        tang = np.array([np.cos(psi), np.sin(psi), 0.0]); tip = b
    else:
        th = np.linspace(psi, psi + sweep, 48)
        pts = x0 + radius * np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], axis=1)
        p.add_mesh(pv.Spline(pts, 96).tube(radius=tube_r), color=color, opacity=opacity, smooth_shading=True)
        tang = np.sign(sweep) * np.array([-np.sin(th[-1]), np.cos(th[-1]), 0.0]); tip = pts[-1]
    p.add_mesh(pv.Cone(center=tip + 0.06 * tang, direction=tang, height=0.14, radius=0.055, resolution=24),
               color=color, opacity=opacity, smooth_shading=True)


def render_view(ep, t, size, pv, label=None, color=C_A):
    p = pv.Plotter(off_screen=True, window_size=size, lighting="three lights")
    p.set_background("#f4f6f9", top="#dfe6ee")
    R, x0 = add_glider(p, ep, t, pv)
    psi = float(ep["yaw"][t])
    add_yaw_arc(p, pv, x0, psi, ep["yaw_ref"][t], 1.30, C_REF, opacity=0.85)          # commanded
    add_yaw_arc(p, pv, x0, psi, ep["yaw_rate"][t], 1.50, color)                        # actual
    # ground reference grid 12 m below the glider (moves with it in discrete 2 m cells, so motion is visible)
    gz = x0[2] - 12.0; cx, cy = np.floor(x0[0] / 2) * 2, np.floor(x0[1] / 2) * 2
    plane = pv.Plane(center=(cx, cy, gz), direction=(0, 0, 1), i_size=60, j_size=60, i_resolution=30, j_resolution=30)
    p.add_mesh(plane, color="#c8d0d9", style="wireframe", line_width=0.5, opacity=0.35)
    yaw = ep["yaw"][t]; back = np.array([-np.cos(yaw), -np.sin(yaw), 0.0])
    side = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    cam = x0 + 2.6 * back + 1.5 * side + np.array([0, 0, 2.3])            # elevated three-quarter chase view
    p.camera_position = [tuple(cam), tuple(x0), (0, 0, 1)]                 # (planform, sweep and dihedral visible)
    p.camera.view_angle = 34
    img = p.screenshot(return_img=True); p.close()
    return img


def make_video(name, panels, fps=25, quick=False, title="", t0=0):
    """panels: list of (label, episode, colour)."""
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    pv.OFF_SCREEN = True
    T = max(len(ep["yaw_rate"]) for _, ep, _ in panels)
    if quick: T = min(T, t0 + 12)
    n = len(panels); W, H = 1920, 1080
    ncol = 1 if n <= 2 else 2; nrow = int(np.ceil(n / ncol))
    view_w = 1150
    vsize = (int(view_w / ncol), int(940 / nrow))
    tmp = tempfile.mkdtemp(prefix=f"vid_{name}_")
    t_axis = np.arange(T) * DT
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    gs = fig.add_gridspec(nrow * 3, ncol + 2, width_ratios=[view_w / ncol] * ncol + [440, 330], left=0.01, right=0.985,
                          top=0.93, bottom=0.07, wspace=0.12, hspace=0.9)
    vax = []
    for k in range(n):
        r, c = divmod(k, ncol)
        a = fig.add_subplot(gs[r * 3:(r + 1) * 3, c]); a.axis("off"); vax.append(a)
    ax_r = fig.add_subplot(gs[0:nrow, ncol:]); ax_a = fig.add_subplot(gs[nrow:2 * nrow, ncol:])
    ax_u = fig.add_subplot(gs[2 * nrow:3 * nrow, ncol:])
    ref = panels[0][1]["yaw_ref"]
    ax_r.plot(t_axis[:len(ref)], ref[:T], color=C_REF, lw=1.2, ls="--", label="reference")
    for lab, ep, col in panels:
        m = min(T, len(ep["yaw_rate"]))
        ax_r.plot(t_axis[:m], ep["yaw_rate"][:m], color=col, lw=1.4, label=lab)
        ax_a.plot(t_axis[:m], np.degrees(ep["roll"][:m]), color=col, lw=1.2)
        ax_a.plot(t_axis[:m], np.degrees(ep["pitch"][:m]), color=col, lw=1.0, ls=":")
        a = ep["action"][:m]
        ax_u.plot(t_axis[:m], 0.5 * (a[:, 0] - a[:, 3]), color=col, lw=1.2)
    ax_r.set_ylabel("yaw rate (rad/s)"); ax_a.set_ylabel("roll — / pitch ··· (deg)")
    ax_u.set_ylabel("diff. sweep cmd (m)"); ax_u.set_xlabel("time (s)")
    ax_r.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=min(len(panels) + 1, 4),
                borderaxespad=0.2)   # above the axes, never over the data
    for a in (ax_r, ax_a, ax_u):
        a.set_xlim(0, T * DT); a.spines[["top", "right"]].set_visible(False)
    cursors = [a.axvline(0, color="#999999", lw=0.8) for a in (ax_r, ax_a, ax_u)]
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold")
    stamp = fig.text(0.985, 0.965, "", ha="right", fontsize=12)
    ims = [None] * n; labs = []; gauges = []
    fs = 11 if n < 3 else 8
    from matplotlib.lines import Line2D
    for k, (lab, ep, col) in enumerate(panels):
        from matplotlib.patches import Rectangle
        vax[k].add_patch(Rectangle((0.012, 0.915), 0.014, 0.06, transform=vax[k].transAxes, color=col, zorder=5))
        labs.append(vax[k].text(0.034, 0.97, "", transform=vax[k].transAxes, va="top", fontsize=12 if n < 5 else 10,
                                color="#1a1a1a", fontweight="bold", zorder=6))
        # turn-rate gauge (left turn drawn on the left: x = -r), commanded vs actual yaw rate
        g = vax[k].inset_axes([0.03, 0.05, 0.36, 0.12]); g.set_xlim(-0.95, 0.95); g.set_ylim(-1.2, 1.2)
        g.patch.set_alpha(0.0)
        for sp in g.spines.values(): sp.set_visible(False)
        g.set_yticks([]); g.set_xticks([-0.6, -0.3, 0, 0.3, 0.6]); g.set_xticklabels(["0.6", "0.3", "0", "0.3", "0.6"], fontsize=fs - 2)
        g.tick_params(axis="x", length=3, pad=1, colors="#4d4d4d")
        g.axhline(0, xmin=0.03, xmax=0.97, color="#7a7a7a", lw=1.2)
        g.text(-0.95, 0.55, "\u2190 left", fontsize=fs - 1, color="#4d4d4d", va="center", ha="left")
        g.text(0.95, 0.55, "right \u2192", fontsize=fs - 1, color="#4d4d4d", va="center", ha="right")
        mt, = g.plot([0], [0.45], marker="v", ms=fs + 1, color=C_REF, ls="none")
        ma, = g.plot([0], [-0.45], marker="^", ms=fs + 1, color=col, ls="none")
        g.legend([Line2D([], [], marker="v", ls="none", color=C_REF, ms=fs - 1), Line2D([], [], marker="^", ls="none", color=col, ms=fs - 1)],
                 ["command", "actual"], loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=fs - 2,
                 handletextpad=0.2, borderaxespad=0.0)
        cmd = vax[k].text(0.03, 0.215, "", transform=vax[k].transAxes, fontsize=fs, color="#1a1a1a", va="bottom", zorder=6)
        gauges.append((mt, ma, cmd))
    fig.text(0.01, 0.040, "3-D view (chase camera). Red spheres: the wingtip points whose positions the controller commands. "
             "Grid: ground, 12 m below.", fontsize=10, color="#4d4d4d", ha="left", va="bottom")
    fig.text(0.01, 0.014, "Arcs: heading change over the next 2 s if the yaw rate stayed constant \u2014 dark grey at the commanded "
             "rate, coloured at the actual rate.", fontsize=10, color="#4d4d4d", ha="left", va="bottom")
    t_start = t0 if quick else 0
    for t in range(t_start, T):
        for k, (lab, ep, col) in enumerate(panels):
            tt = min(t, len(ep["yaw_rate"]) - 1)
            img = render_view(ep, tt, vsize, pv, color=col)
            rr, r = float(ep["yaw_ref"][tt]), float(ep["yaw_rate"][tt])
            mt, ma, cmd = gauges[k]
            mt.set_xdata([-np.clip(rr, -0.92, 0.92)]); ma.set_xdata([-np.clip(r, -0.92, 0.92)])
            side = lambda v: "left" if v > 0 else "right"
            ctext = "hold heading" if abs(rr) < 0.02 else f"yaw {side(rr)} {abs(rr):.2f} rad/s"
            atext = "0.00 rad/s" if abs(r) < 0.005 else f"{side(r)} {abs(r):.2f} rad/s"
            cmd.set_text(f"Command: {ctext}     actual: {atext}")
            if ims[k] is None: ims[k] = vax[k].imshow(img)
            else: ims[k].set_data(img)
            status = "  — episode terminated" if (t >= len(ep["yaw_rate"]) - 1 and ep["failed"]) else ""
            labs[k].set_text(f"{lab}   RMS@H {ep['rms_h']:.3f} rad/s{status}")
        for cu in cursors: cu.set_xdata([t * DT, t * DT])
        stamp.set_text(f"t = {t * DT:4.2f} s   (simulation, real time)")
        fig.savefig(os.path.join(tmp, f"f{t - t_start:04d}.png"), dpi=100)
    plt.close(fig)
    out = os.path.join(OUT, f"{name}.mp4"); os.makedirs(OUT, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-vf", "scale=1920:1080", out], check=True)
    shutil.rmtree(tmp)
    print("[video]", out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*"); ap.add_argument("--quick", action="store_true"); ap.add_argument("--t0", type=int, default=0)
    a = ap.parse_args()
    info = json.load(open(os.path.join(DATA, "episode.json")))
    teacher = load("sac_teacher")
    smooth = load("sac_smooth") if os.path.exists(os.path.join(DATA, "sac_smooth.npz")) else None
    main_pol, main_lab = (smooth, "SAC-S (smoothness fine-tuned)") if smooth is not None else (teacher, "SAC (trained policy)")
    cl = sorted(glob.glob(os.path.join(DATA, "classical_*.npz")))
    jobs = {"S1": ("S1_sac_policy", [(main_lab, main_pol, C_A)],
                   f"S1  Trained {'SAC-S' if smooth is not None else 'SAC'} policy, nominal evaluation episode (seed {info['eval_seed']})")}
    if cl:
        tag = os.path.basename(cl[0])[:-4]; name = tag.replace("classical_", "")
        short = "SAC-S" if smooth is not None else "SAC"
        jobs["S2"] = ("S2_sac_vs_classical", [(short, main_pol, C_A), (f"{name} (best tuned classical)", load(tag), C_B)],
                      f"S2  {short} vs best tuned classical controller, identical episode")
    if smooth is not None:
        jobs["S5"] = ("S5_sac_vs_sac_smooth", [("SAC (1M steps)", teacher, "#e08a1e"), ("SAC-S (+ smoothness fine-tuning)", smooth, C_A)],
                      "S5  SAC before vs after smoothness fine-tuning, identical episode")
    ck = sorted(glob.glob(os.path.join(DATA, "sac_seed7_*.npz")))
    if ck:
        cols = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
        pan = []
        for k, f in enumerate(ck):
            st = int(os.path.basename(f)[:-4].split("_")[-1])
            pan.append((f"{st // 1000}k steps", load(os.path.basename(f)[:-4]), cols[k % len(cols)]))
        jobs["S3"] = ("S3_training_progress", pan, "S3  Training progress: SAC (seed 7) checkpoints on the same episode")
    if os.path.exists(os.path.join(DATA, "kan_student.npz")):
        jobs["S4"] = ("S4_sac_vs_kan", [("SAC teacher", teacher, C_A), ("KAN student", load("kan_student"), C_C)],
                      "S4  SAC teacher vs distilled KAN student, identical episode")
    for k, (name, panels, title) in jobs.items():
        if a.only and k not in a.only: continue
        make_video(name, panels, quick=a.quick, title=title, t0=a.t0)


if __name__ == "__main__":
    main()
