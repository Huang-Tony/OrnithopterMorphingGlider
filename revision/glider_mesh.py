"""Glider surface geometry (numpy only, no simulator imports): used by the 3D renderer for figures and videos.

Body frame as in the simulator: x forward, y toward the 'R' wing, z up; spar root P0 at the origin.
Each wing is lofted along its cubic-Bezier spar (quarter-chord line) with an E387-like section
(NACA-form camber line, 3.8 % camber at 40 % chord, 9.1 % thickness) oriented by the aerodynamic proxy's
chord vector c(u) (root: body x axis; tip: x axis projected normal to the local span direction).
"""

import numpy as np

CHORD = 0.15
X_HAT = np.array([1.0, 0.0, 0.0])


def bezier(P, u):
    P = np.asarray(P, float); u = np.asarray(u, float)[:, None]
    return (1 - u) ** 3 * P[0] + 3 * (1 - u) ** 2 * u * P[1] + 3 * (1 - u) * u ** 2 * P[2] + u ** 3 * P[3]


def bezier_tangent(P, u):
    P = np.asarray(P, float); u = np.asarray(u, float)[:, None]
    return 3 * (1 - u) ** 2 * (P[1] - P[0]) + 6 * (1 - u) * u * (P[2] - P[1]) + 3 * u ** 2 * (P[3] - P[2])


def airfoil(n=48, m=0.038, p=0.40, t=0.091):
    beta = np.linspace(0, np.pi, n); x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
    dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x)); th = np.arctan(dyc)
    xu, zu = x - yt * np.sin(th), yc + yt * np.cos(th); xl, zl = x + yt * np.sin(th), yc - yt * np.cos(th)
    xs = np.concatenate([xu[::-1], xl[1:]]); zs = np.concatenate([zu[::-1], zl[1:]])
    return xs, zs     # closed loop TE(upper) -> LE -> TE(lower), in chord units


def section_frames(P, u):
    pos = bezier(P, u); tan = bezier_tangent(P, u)
    s = tan / np.linalg.norm(tan, axis=1, keepdims=True)
    c_tip = X_HAT - s[:, [0]] * s
    uu = np.asarray(u, float)[:, None]
    c = (1 - uu) * X_HAT + uu * c_tip; c /= np.linalg.norm(c, axis=1, keepdims=True)
    n = np.cross(c, s); n /= np.linalg.norm(n, axis=1, keepdims=True)
    if float(np.asarray(P)[3][1]) < 0: n = -n          # upward normal on the y<0 wing (as in the fixed model)
    return pos, c, n


def wing_grid(P, n_span=48, n_af=48):
    """Return (n_span, n_af_closed, 3) array of skin points."""
    u = np.linspace(0.0, 1.0, n_span); pos, c, n = section_frames(P, u)
    xs, zs = airfoil(n_af)
    X = pos[:, None, :] + ((0.25 - xs)[None, :, None] * CHORD) * c[:, None, :] + (zs[None, :, None] * CHORD) * n[:, None, :]
    return X


def fuselage_grid(n_x=60, n_t=40, x_tail=-0.34, x_nose=0.24, r_max=0.030, z_off=-0.018):
    xi = np.linspace(0, 1, n_x)
    x = x_tail + (x_nose - x_tail) * xi
    nose = np.clip((x - 0.10) / (x_nose - 0.10), 0, 1)
    r = r_max * np.where(x > 0.10, np.sqrt(np.clip(1 - nose ** 2, 0, 1)), 1.0)
    tail = np.clip((0.0 - x) / (0.0 - x_tail), 0, 1)
    r = r * np.where(x < 0.0, 1 - 0.80 * tail ** 1.3, 1.0)
    th = np.linspace(0, 2 * np.pi, n_t)
    Y = r[:, None] * np.cos(th)[None]; Z = z_off + 0.85 * r[:, None] * np.sin(th)[None]
    Xg = np.repeat(x[:, None], n_t, axis=1)
    return np.stack([Xg, Y, Z], axis=-1)


def euler_to_R(roll, pitch, yaw):
    """Body->world rotation for the simulator's ZYX Euler angles (quat_to_euler_xyz)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx
