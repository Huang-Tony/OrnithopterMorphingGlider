#!/usr/bin/env python3
"""Render the glider in 3D (PyVista/VTK, off-screen). Runs in a separate environment with pyvista:
    <render_venv>/bin/python revision/render_glider.py
Reads revision_runs/analysis/platform_configs.json; writes transparent PNG renders + 2-D anchor points
(screen coordinates of labelled 3-D points) to revision_runs/paper_figures/renders/.
"""

import json
import os
import sys

import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from glider_mesh import wing_grid, fuselage_grid, bezier  # noqa: E402

RUNS = os.environ.get("REVISION_RUNS_DIR", os.path.join(os.path.dirname(HERE), "revision_runs"))
OUT = os.path.join(RUNS, "paper_figures", "renders"); os.makedirs(OUT, exist_ok=True)
pv.OFF_SCREEN = True

SKIN = "#c9d3de"; SKIN_EDGE = "#5d6b79"; SPAR = "#1f5fae"; FUS = "#b9bec5"; CP = "#2a78d6"; ACC = "#d9542b"


def grid_surface(X):
    g = pv.StructuredGrid(X[..., 0], X[..., 1], X[..., 2])
    return g.extract_surface().triangulate().compute_normals(auto_orient_normals=False, split_vertices=False)


def add_glider(p, cfg, *, skin_opacity=0.62, show_cp=False, spar_r=0.0055, n_ribs=11):
    for side in ("R", "L"):
        P = np.array(cfg[side])
        X = wing_grid(P, n_span=61)
        p.add_mesh(grid_surface(X), color=SKIN, opacity=skin_opacity, smooth_shading=True,
                   specular=0.25, specular_power=20, ambient=0.25, diffuse=0.75)
        for k in np.linspace(0, X.shape[0] - 1, n_ribs).round().astype(int):          # ribs
            p.add_mesh(pv.lines_from_points(np.vstack([X[k], X[k][:1]])), color=SKIN_EDGE, line_width=1.0, opacity=0.9)
        i_le = int(np.argmin(np.abs(np.arange(X.shape[1]) - X.shape[1] // 2)))
        for j in (0, i_le):                                                              # trailing / leading edge
            p.add_mesh(pv.Spline(X[:, j, :], 200), color=SKIN_EDGE, line_width=1.6)
        C = bezier(P, np.linspace(0, 1, 120))
        p.add_mesh(pv.Spline(C, 240).tube(radius=spar_r, n_sides=24), color=SPAR, smooth_shading=True,
                   specular=0.4, ambient=0.2)
        if show_cp is True or (isinstance(show_cp, (set, tuple, list)) and side in show_cp):
            p.add_mesh(pv.lines_from_points(P), color=SPAR, line_width=1.5, opacity=0.9)
            for k, q in enumerate(P):
                p.add_mesh(pv.Sphere(radius=0.0095 if k else 0.012, center=q), color=CP if k else "#0d3b74",
                           smooth_shading=True, specular=0.3)
    p.add_mesh(grid_surface(fuselage_grid()), color=FUS, smooth_shading=True, specular=0.35, ambient=0.25)


def world_to_px(p, pts):
    import vtk
    ren = p.renderer; coord = vtk.vtkCoordinate(); coord.SetCoordinateSystemToWorld()
    W, H = p.window_size; out = []
    for q in pts:
        coord.SetValue(*map(float, q)); x, y = coord.GetComputedDoubleDisplayValue(ren)
        out.append([x / W, 1 - y / H])       # normalised image coords (origin top-left)
    return out


def render(cfg, name, camera, *, show_cp=False, anchors=None, size=(2400, 1500), zoom=1.0, skin_opacity=0.55,
           parallel=False, extras=None, view_angle=None):
    p = pv.Plotter(off_screen=True, window_size=size, lighting="three lights")
    p.set_background("white"); p.enable_anti_aliasing("ssaa"); p.enable_depth_peeling(number_of_peels=8)
    add_glider(p, cfg, show_cp=show_cp, skin_opacity=skin_opacity)
    if extras: extras(p)
    p.camera_position = camera
    if view_angle: p.camera.view_angle = view_angle
    if parallel:
        p.enable_parallel_projection(); p.camera.parallel_scale = 0.58      # half-height (m); fits the 2 m span
    else:
        p.camera.zoom(zoom)
    p.render()
    out = {}
    if anchors:
        px = world_to_px(p, list(anchors.values()))
        out = {k: v for k, v in zip(anchors.keys(), px)}
    p.screenshot(os.path.join(OUT, f"{name}.png"), transparent_background=True)
    p.close()
    return out


def main():
    data = json.load(open(os.path.join(RUNS, "analysis", "platform_configs.json")))
    cfgs = data["configs"]
    hero_cam = [(3.3, 2.9, 2.35), (-0.05, 0.05, 0.02), (0, 0, 1)]     # elevated front three-quarter view
    H = cfgs["hero"]; R, L = np.array(H["R"]), np.array(H["L"])
    tip = R[3]; tri0 = np.array([0.50, -0.42, -0.30]); La = 0.18
    def extras(p):
        for d in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):        # commanded tip degrees of freedom
            p.add_mesh(pv.Arrow(start=tip, direction=d, scale=0.17, tip_length=0.28, tip_radius=0.07, shaft_radius=0.025),
                       color=ACC, smooth_shading=True)
        for d in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):        # body axes
            p.add_mesh(pv.Arrow(start=tri0, direction=d, scale=La, tip_length=0.25, tip_radius=0.06, shaft_radius=0.02),
                       color="#4d4d4d", smooth_shading=True)
    anchors = {"P0": R[0], "R_P1": R[1], "R_P2": R[2], "R_P3": R[3],
               "fuselage_nose": [0.20, 0.0, -0.025], "skin_L": bezier(L, [0.55])[0] + np.array([-0.06, 0, 0.0]),
               "spar_R": bezier(R, [0.80])[0], "tip_dx": tip + [0.19, 0, 0], "tip_dy": tip + [0, 0.19, 0],
               "tip_dz": tip + [0, 0, 0.19], "ax_x": tri0 + [La + 0.03, 0, 0], "ax_y": tri0 + [0, La + 0.03, 0],
               "ax_z": tri0 + [0, 0, La + 0.03], "ax_o": tri0}
    a = render(H, "hero", hero_cam, show_cp={"R"}, anchors=anchors, zoom=1.55, extras=extras, view_angle=24, size=(3200, 2000))
    front = [(3.0, 0.0, 0.10), (0.0, 0.0, 0.10), (0, 0, 1)]         # looking aft, parallel projection
    top = [(0.0, 0.0, 3.0), (0.0, 0.0, 0.0), (1, 0, 0)]             # plan view, nose up (+y wing on the left)
    for nm, cam, tag in (("neutral", front, "front"), ("diff_sweep", top, "top"), ("diff_dihedral", front, "front")):
        render(cfgs[nm], f"cfg_{nm}_{tag}", cam, size=(2000, 900), zoom=1.0, skin_opacity=0.72, show_cp=False, parallel=True)
    json.dump({"hero": a}, open(os.path.join(OUT, "anchors.json"), "w"), indent=1)
    print("rendered:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
