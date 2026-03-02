"""Quaternion utilities for 6DOF attitude dynamics."""

import math
import numpy as np


def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / max(n, 1e-12)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_to_rotmat_body_to_world(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w),     2*(x*z+y*w)],
                     [2*(x*y+z*w),     1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),     2*(y*z+x*w),     1 - 2*(x*x+y*y)]])


def quat_integrate_body_rates(q, omega_body, dt):
    p, qq, r = omega_body
    omega_norm = math.sqrt(p*p + qq*qq + r*r)
    if omega_norm < 1e-12:
        return quat_normalize(q)
    half_angle = 0.5 * omega_norm * dt
    s = math.sin(half_angle) / omega_norm
    dq = np.array([math.cos(half_angle), s*p, s*qq, s*r])
    return quat_normalize(quat_mul(q, dq))


def quat_to_euler_xyz(q):
    w, x, y, z = q
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2*(w*y - z*x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw
