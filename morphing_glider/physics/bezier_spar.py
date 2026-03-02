import numpy as np

from morphing_glider.config import L_FIXED


class RealTimeBezierSpar:
    """Cubic Bezier spar with approximate fixed-length + curvature-energy solver.
    Proxy equilibrium solver (NOT a structural FEM)."""
    def __init__(self, p0, p3_target, p1_guess, p2_guess):
        self.p0 = np.array(p0, dtype=float); self.p3 = np.array(p3_target, dtype=float)
        self.p1 = np.array(p1_guess, dtype=float); self.p2 = np.array(p2_guess, dtype=float)
        self.learning_rate = 0.04; self.iterations = 10; self.lock_z = False
        self._last_len = float("nan"); self._last_energy = float("nan")

    def evaluate(self, u):
        u = np.asarray(u, dtype=float)
        return ((1-u)**3)*self.p0 + 3*((1-u)**2)*u*self.p1 + 3*(1-u)*(u**2)*self.p2 + (u**3)*self.p3

    def tangent(self, u):
        u = np.asarray(u, dtype=float)
        return 3*((1-u)**2)*(self.p1-self.p0) + 6*(1-u)*u*(self.p2-self.p1) + 3*(u**2)*(self.p3-self.p2)

    def _get_len_energy(self, p1, p2, n_samples=18):
        t = np.linspace(0, 1, int(n_samples)).reshape(-1, 1)
        points = ((1-t)**3)*self.p0 + 3*((1-t)**2)*t*p1 + 3*(1-t)*(t**2)*p2 + (t**3)*self.p3
        dists = np.sqrt(np.sum((points[1:]-points[:-1])**2, axis=1))
        current_len = float(np.sum(dists))
        energy = float(np.sum((p1-self.p0)**2) + np.sum((p2-p1)**2) + np.sum((self.p3-p2)**2))
        return current_len, energy

    def length_and_energy(self):
        return self._get_len_energy(self.p1, self.p2)

    def solve_shape(self, *, iterations=None, w_len=55.0, w_energy=1.0, w_bio=2.0, eps=1e-3, grad_clip=0.6):
        target_len = float(L_FIXED); iters = self.iterations if iterations is None else int(iterations)
        if self.lock_z:
            self.p1[2]=0.0; self.p2[2]=0.0; self.p3[2]=0.0; w_bio=0.0
        for _ in range(int(iters)):
            cl, ce = self._get_len_energy(self.p1, self.p2)
            self._last_len = float(cl); self._last_energy = float(ce)
            bc = float(w_energy)*ce + float(w_len)*(cl-target_len)**2 - float(w_bio)*(self.p1[2]+self.p2[2])
            g1 = np.zeros(3); g2 = np.zeros(3)
            dims = range(2) if self.lock_z else range(3)
            for i in dims:
                p1t = self.p1.copy(); p1t[i] += float(eps); l,e = self._get_len_energy(p1t, self.p2)
                g1[i] = (float(w_energy)*e + float(w_len)*(l-target_len)**2 - float(w_bio)*(p1t[2]+self.p2[2]) - bc) / float(eps)
            for i in dims:
                p2t = self.p2.copy(); p2t[i] += float(eps); l,e = self._get_len_energy(self.p1, p2t)
                g2[i] = (float(w_energy)*e + float(w_len)*(l-target_len)**2 - float(w_bio)*(self.p1[2]+p2t[2]) - bc) / float(eps)
            self.p1 -= float(self.learning_rate) * np.clip(g1, -grad_clip, +grad_clip)
            self.p2 -= float(self.learning_rate) * np.clip(g2, -grad_clip, +grad_clip)
            if self.lock_z: self.p1[2]=0.0; self.p2[2]=0.0; self.p3[2]=0.0
        return self.p1, self.p2

    def solve_to_convergence(self, *, max_total_iters=80, chunk_iters=12, tol_len=1e-3):
        for _ in range(int(max_total_iters // max(1, chunk_iters))):
            self.solve_shape(iterations=int(chunk_iters))
            cl, _ = self.length_and_energy()
            if abs(cl - float(L_FIXED)) <= float(tol_len): break
