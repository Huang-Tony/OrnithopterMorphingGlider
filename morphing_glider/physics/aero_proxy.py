import numpy as np

from morphing_glider.config import L_FIXED, WING_CHORD


class AeroProxy3D:
    """Panel model with 3D force directions. Returns net force, moment, diagnostics."""
    def __init__(self, *, num_panels=12, include_omega_cross=True):
        self.N = int(num_panels); self.include_omega_cross = bool(include_omega_cross)
        self.u = ((np.arange(self.N, dtype=float)+0.5)/self.N).reshape(-1,1)
        self.area = (L_FIXED/self.N)*WING_CHORD; self.x_axis = np.array([1.0,0.0,0.0])

    def calculate_forces(self, spar, *, v_rel_body, omega_body, phys):
        pos = spar.evaluate(self.u); tan = spar.tangent(self.u)
        t_norm = np.linalg.norm(tan, axis=1, keepdims=True); span = tan/(t_norm+1e-9)
        dot = span[:,[0]]; c_tip = self.x_axis - dot*span
        c = (1.0-self.u)*self.x_axis + self.u*c_tip; c = c/(np.linalg.norm(c, axis=1, keepdims=True)+1e-9)
        n = np.cross(c, span); n = n/(np.linalg.norm(n, axis=1, keepdims=True)+1e-9)
        omega = np.asarray(omega_body, dtype=float).reshape(3); v_rel = np.asarray(v_rel_body, dtype=float).reshape(3)
        if self.include_omega_cross: v_local = v_rel[None,:] + np.cross(omega[None,:], pos)
        else: v_local = np.broadcast_to(v_rel[None,:], pos.shape).copy()
        speed = np.linalg.norm(v_local, axis=1)+1e-9; v_hat = v_local/speed[:,None]
        alpha = -np.arctan2(np.sum(v_hat*n, axis=1), np.sum(v_hat*c, axis=1))
        alpha = alpha + float(phys.get("alpha0", 0.0))
        alpha_clip = float(phys["alpha_clip"]); alpha = np.clip(alpha, -alpha_clip, +alpha_clip)
        cl_alpha = float(phys["cl_alpha"]); cl_max = float(phys["cl_max"])
        CL = cl_max * np.tanh(cl_alpha*alpha/max(1e-6, cl_max))
        cd0 = float(phys["cd0"]); k_ind = float(phys["k_induced"])
        stall = np.maximum(0.0, np.abs(alpha)-float(phys["alpha_stall"])) / max(1e-6, alpha_clip)
        CD = cd0 + k_ind*(CL**2) + float(phys["cd_stall"])*(stall**2)
        rho = float(phys["rho"]); q_dyn = 0.5*rho*(speed**2)
        n_perp = n - (np.sum(n*v_hat, axis=1, keepdims=True))*v_hat
        lift_dir = n_perp/(np.linalg.norm(n_perp, axis=1, keepdims=True)+1e-9)
        ls = float(phys.get("lift_scale", 1.0))
        lift = ls*q_dyn*self.area*CL; drag = q_dyn*self.area*CD
        F = lift[:,None]*lift_dir + drag[:,None]*(-v_hat)
        total_force = np.sum(F, axis=0); moments = np.cross(pos, F); total_moment = np.sum(moments, axis=0)
        power = -np.sum(np.sum(F*v_local, axis=1)); power_loss = float(max(0.0, power))
        diag = dict(total_drag_force=float(np.sum(drag)), total_lift_force=float(np.sum(lift)),
                     mean_alpha=float(np.mean(alpha)), mean_abs_alpha=float(np.mean(np.abs(alpha))),
                     mean_speed=float(np.mean(speed)), power_loss=float(power_loss))
        return total_force, total_moment, diag
