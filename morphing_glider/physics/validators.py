import warnings
from typing import Any, Dict, Optional

import numpy as np

from morphing_glider.config import (
    WING_CHORD, L_FIXED, EVAL_AERO_PANELS, BEZIER_ITERS_EVAL,
    DZ_RANGE, WING_AREA_TOTAL, EB_YOUNGS_MODULUS, EB_SPAR_THICKNESS,
)
from morphing_glider.physics.bezier_spar import RealTimeBezierSpar
from morphing_glider.physics.aero_proxy import AeroProxy3D


def _biot_savart_segment(A: np.ndarray, B: np.ndarray, P: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Velocity induced at P by unit-circulation vortex segment from A to B.

    Args:
        A: Start point of vortex segment (3,).
        B: End point of vortex segment (3,).
        P: Field point (3,).
        eps: Regularization epsilon.

    Returns:
        Induced velocity vector (3,).

    References:
        [KATZ_PLOTKIN_2001] Low-Speed Aerodynamics, 2nd ed. Cambridge.
    """
    r0 = B - A; r1 = P - A; r2 = P - B
    r1m = np.linalg.norm(r1); r2m = np.linalg.norm(r2)
    cross = np.cross(r1, r2); cs = np.dot(cross, cross)
    if cs < eps * eps or r1m < eps or r2m < eps:
        return np.zeros(3, dtype=float)
    return (1.0 / (4.0 * np.pi)) * cross / cs * np.dot(r0, r1 / r1m - r2 / r2m)


class VortexLatticeReference:
    """Simplified 3D vortex lattice method for a flat rectangular wing.

    Uses horseshoe vortex elements with 1 chordwise panel (equivalent to
    corrected lifting-line). Provides CL, CD_induced, and yaw moment
    coefficient as reference for AeroProxy3D validation.

    Args:
        num_spanwise: Number of spanwise panels (full span).
        chord: Wing chord [m].
        half_span: Wing semi-span [m].

    Returns:
        CL, CD_induced, Cm_yaw via solve() method.

    References:
        [KATZ_PLOTKIN_2001] Low-Speed Aerodynamics.
        [ANDERSON_2017] Fundamentals of Aerodynamics, 6th ed.
    """
    # [HW_VALIDATION_REQUIRED: compare against wind tunnel force balance data at Re ~ 1e5]

    def __init__(self, num_spanwise: int = 12, chord: float = WING_CHORD, half_span: float = L_FIXED):
        self.N = max(4, int(num_spanwise))
        self.c = float(chord)
        self.b = float(2.0 * half_span)
        self.S = float(self.b * self.c)
        dy = self.b / self.N
        self.dy = dy
        self.y_cp = np.array([(i + 0.5) * dy - self.b / 2.0 for i in range(self.N)])
        self.y_v1 = np.array([i * dy - self.b / 2.0 for i in range(self.N)])
        self.y_v2 = np.array([(i + 1) * dy - self.b / 2.0 for i in range(self.N)])
        x_bv = self.c * 0.25
        x_cp = self.c * 0.75
        x_far = 50.0 * self.b
        self.x_bv = x_bv
        self.x_cp = x_cp
        self.x_far = x_far
        self._build_aic()

    def _build_aic(self) -> None:
        N = self.N
        AIC = np.zeros((N, N), dtype=float)
        for i in range(N):
            P = np.array([self.x_cp, self.y_cp[i], 0.0])
            for j in range(N):
                A_bv = np.array([self.x_bv, self.y_v1[j], 0.0])
                B_bv = np.array([self.x_bv, self.y_v2[j], 0.0])
                v_bound = _biot_savart_segment(A_bv, B_bv, P)
                A_trail_L = np.array([self.x_far, self.y_v1[j], 0.0])
                v_trail_L = _biot_savart_segment(A_trail_L, A_bv, P)
                B_trail_R = np.array([self.x_far, self.y_v2[j], 0.0])
                v_trail_R = _biot_savart_segment(B_bv, B_trail_R, P)
                w_total = v_bound[2] + v_trail_L[2] + v_trail_R[2]
                AIC[i, j] = w_total
        self.AIC = AIC

    def solve(self, alpha_rad: float, V_inf: float = 15.0, rho: float = 1.225,
              twist_distribution: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Solve for aerodynamic coefficients.

        Args:
            alpha_rad: Angle of attack [rad].
            V_inf: Freestream velocity [m/s].
            rho: Air density [kg/m^3].
            twist_distribution: Per-panel twist angles [rad], shape (N,). Added to alpha.

        Returns:
            Dict with CL, CD_induced, Cm_yaw, and Gamma distribution.

        References:
            [ANDERSON_2017] Fundamentals of Aerodynamics.
        """
        N = self.N
        alpha_local = np.full(N, float(alpha_rad))
        if twist_distribution is not None:
            alpha_local = alpha_local + np.asarray(twist_distribution, dtype=float)[:N]
        rhs = -V_inf * np.sin(alpha_local)
        try:
            Gamma = np.linalg.solve(self.AIC, rhs)
        except np.linalg.LinAlgError:
            Gamma = np.zeros(N)
        L_panels = rho * V_inf * Gamma * self.dy
        CL = float(np.sum(L_panels)) / (0.5 * rho * V_inf**2 * self.S)
        w_induced = self.AIC @ Gamma
        alpha_induced = -w_induced / max(V_inf, 1e-9)
        D_panels = -rho * w_induced * Gamma * self.dy
        CD_i = float(np.sum(D_panels)) / (0.5 * rho * V_inf**2 * self.S)
        M_yaw = float(np.sum(D_panels * self.y_cp))
        Cm_yaw = M_yaw / (0.5 * rho * V_inf**2 * self.S * self.b)
        return {"CL": CL, "CD_induced": max(0.0, CD_i), "Cm_yaw": Cm_yaw,
                "Gamma": Gamma.copy(), "L_panels": L_panels.copy(),
                "M_yaw_Nm": M_yaw}


class EulerBernoulliBeamReference:
    """Euler-Bernoulli cantilever beam model for spar deflection validation.

    Models the morphing spar as a cantilever beam under tip load F:
        $v(x) = (F x^2 / (6EI)) (3L - x)$

    Args:
        L: Beam length [m].
        E: Young's modulus [Pa].
        b: Beam width [m] (chord direction).
        h: Beam thickness [m].

    Returns:
        Deflection profile and bending energy via compute() method.

    References:
        [GERE_GOODNO_2018] Mechanics of Materials, 9th ed.
    """
    # [CALIBRATION_REQUIRED: measure actual spar EI from 3-point bend test]

    def __init__(self, L: float = L_FIXED, E: float = EB_YOUNGS_MODULUS,
                 b: float = WING_CHORD, h: float = EB_SPAR_THICKNESS):
        self.L = float(L)
        self.E = float(E)
        self.I = float(b * h**3 / 12.0)  # [m^4]
        self.EI = float(self.E * self.I)   # [N·m^2]

    def deflection(self, x: np.ndarray, F: float) -> np.ndarray:
        """Compute beam deflection at positions x under tip load F.

        Args:
            x: Array of positions along beam [0, L] in meters.
            F: Tip load [N] (positive = upward).

        Returns:
            Deflection array v(x) [m].
        """
        x = np.asarray(x, dtype=float)
        L = self.L; EI = self.EI
        return (F * x**2 / (6.0 * EI)) * (3.0 * L - x)

    def bending_energy(self, F: float) -> float:
        """Compute total bending energy for a tip load F.

        $U = F^2 L^3 / (6 EI)$

        Args:
            F: Tip load [N].

        Returns:
            Bending energy [J].
        """
        return float(F**2 * self.L**3 / (6.0 * self.EI))

    def tip_load_for_deflection(self, delta: float) -> float:
        """Compute tip load F that produces tip deflection delta.

        $delta = F L^3 / (3 EI)$ → $F = 3 EI delta / L^3$

        Args:
            delta: Desired tip deflection [m].

        Returns:
            Required tip load F [N].
        """
        return float(3.0 * self.EI * delta / self.L**3)


def validate_aero_proxy(phys: Dict[str, float], n_alpha: int = 12) -> Dict[str, Any]:
    """Compare AeroProxy3D against VortexLatticeReference across alpha sweep.

    Args:
        phys: Physics parameter dict (nominal).
        n_alpha: Number of angle-of-attack points.

    Returns:
        Dict with pearson_r, rmse, max_deviation, and per-alpha comparison data.

    References:
        [KATZ_PLOTKIN_2001] Low-Speed Aerodynamics.
    """
    alphas = np.linspace(-15.0, 15.0, n_alpha) * np.pi / 180.0
    V0 = float(phys.get("V0", 15.0))
    rho = float(phys.get("rho", 1.225))
    vlm = VortexLatticeReference(num_spanwise=2 * EVAL_AERO_PANELS, chord=WING_CHORD, half_span=L_FIXED)
    aero = AeroProxy3D(num_panels=EVAL_AERO_PANELS, include_omega_cross=True)
    spar_R = RealTimeBezierSpar([0,0,0], [0,+L_FIXED,0], [0,+L_FIXED*0.33,0], [0,+L_FIXED*0.66,0])
    spar_L = RealTimeBezierSpar([0,0,0], [0,-L_FIXED,0], [0,-L_FIXED*0.33,0], [0,-L_FIXED*0.66,0])
    spar_R.iterations = BEZIER_ITERS_EVAL; spar_L.iterations = BEZIER_ITERS_EVAL
    spar_R.solve_to_convergence(); spar_L.solve_to_convergence()
    cl_proxy = []; cl_vlm = []
    for alpha in alphas:
        phys_a = dict(phys); phys_a["alpha0"] = float(alpha)
        v_body = np.array([V0, 0.0, 0.0])
        F_R, _, d_R = aero.calculate_forces(spar_R, v_rel_body=v_body, omega_body=np.zeros(3), phys=phys_a)
        F_L, _, d_L = aero.calculate_forces(spar_L, v_rel_body=v_body, omega_body=np.zeros(3), phys=phys_a)
        lift_total = float(d_R["total_lift_force"] + d_L["total_lift_force"])
        cl_p = lift_total / (0.5 * rho * V0**2 * WING_AREA_TOTAL + 1e-9)
        cl_proxy.append(cl_p)
        res = vlm.solve(alpha, V_inf=V0, rho=rho)
        cl_vlm.append(res["CL"])
    cl_proxy = np.array(cl_proxy); cl_vlm = np.array(cl_vlm)
    mask = np.isfinite(cl_proxy) & np.isfinite(cl_vlm)
    if mask.sum() < 3:
        return {"pearson_r": float("nan"), "rmse": float("nan"), "max_deviation": float("nan")}
    corr = float(np.corrcoef(cl_proxy[mask], cl_vlm[mask])[0, 1]) if mask.sum() > 2 else float("nan")
    rmse_val = float(np.sqrt(np.mean((cl_proxy[mask] - cl_vlm[mask])**2)))
    # Robust relative deviation: use max(|CL_vlm|, 0.15) as denominator
    # to avoid blow-up at near-zero CL (near alpha=0)
    denom_robust = np.maximum(np.abs(cl_vlm[mask]), 0.15)
    max_dev = float(np.max(np.abs(cl_proxy[mask] - cl_vlm[mask]) / denom_robust))
    # CL slope comparison (proxy effective slope vs VLM slope)
    ls = float(phys.get("lift_scale", 1.0))
    if mask.sum() >= 3:
        slope_proxy = float(np.polyfit(alphas[mask], cl_proxy[mask], 1)[0])
        slope_vlm = float(np.polyfit(alphas[mask], cl_vlm[mask], 1)[0])
        slope_ratio = slope_proxy / max(abs(slope_vlm), 1e-9)
    else:
        slope_proxy = slope_vlm = slope_ratio = float("nan")
    print("\n" + "="*80)
    print("[AERO PROXY VALIDATION] AeroProxy3D vs VortexLatticeReference")
    print("="*80)
    print(f"  Alpha range: {np.degrees(alphas[0]):.1f}\u00b0 to {np.degrees(alphas[-1]):.1f}\u00b0 ({n_alpha} pts)")
    print(f"  Pearson r:          {corr:.4f}")
    print(f"  RMSE(CL):           {rmse_val:.4f}")
    print(f"  Max rel. deviation: {max_dev*100:.1f}%  (denom floor=0.15)")
    print(f"  CL slope proxy:     {slope_proxy:.4f} rad^-1")
    print(f"  CL slope VLM:       {slope_vlm:.4f} rad^-1")
    print(f"  Slope ratio:        {slope_ratio:.4f}  (1.0 = perfect)")
    print(f"  lift_scale used:    {ls:.3f}")
    if corr < 0.90 or max_dev > 0.40:
        warnings.warn(f"Aero proxy validation: Pearson r={corr:.3f}, max_dev={max_dev:.3f}. "
                      "Consider recalibrating.", RuntimeWarning)
    return {"pearson_r": corr, "rmse": rmse_val, "max_deviation": max_dev,
            "slope_proxy": slope_proxy, "slope_vlm": slope_vlm, "slope_ratio": slope_ratio,
            "alphas_deg": np.degrees(alphas).tolist(), "cl_proxy": cl_proxy.tolist(), "cl_vlm": cl_vlm.tolist()}


def validate_spar_proxy(n_deflections: int = 8) -> Dict[str, Any]:
    """Compare Bezier spar curvature energy against Euler-Bernoulli bending energy.

    Args:
        n_deflections: Number of tip deflection points to sweep.

    Returns:
        Dict with correlation, rmse, and energy profiles.

    References:
        [GERE_GOODNO_2018] Mechanics of Materials.
    """
    eb = EulerBernoulliBeamReference()
    dz_vals = np.linspace(0.0, DZ_RANGE[1], n_deflections)
    e_bezier = []; e_eb = []
    for dz in dz_vals:
        spar = RealTimeBezierSpar([0,0,0], [0, L_FIXED, float(dz)],
                                  [0, L_FIXED*0.33, 0], [0, L_FIXED*0.66, 0])
        spar.iterations = BEZIER_ITERS_EVAL
        spar.solve_to_convergence()
        _, energy = spar.length_and_energy()
        e_bezier.append(energy)
        F = eb.tip_load_for_deflection(abs(dz))
        e_eb.append(eb.bending_energy(F))
    e_bezier = np.array(e_bezier); e_eb = np.array(e_eb)
    mask = np.isfinite(e_bezier) & np.isfinite(e_eb) & (e_eb > 1e-12)
    corr = float(np.corrcoef(e_bezier[mask], e_eb[mask])[0, 1]) if mask.sum() > 2 else float("nan")
    rmse_val = float(np.sqrt(np.mean((e_bezier[mask] - e_eb[mask])**2))) if mask.any() else float("nan")
    print("\n" + "="*80)
    print("[SPAR PROXY VALIDATION] Bezier energy vs Euler-Bernoulli")
    print("="*80)
    print(f"  Deflections: {n_deflections} pts from 0 to {DZ_RANGE[1]:.3f} m")
    print(f"  Correlation: {corr:.4f}")
    print(f"  RMSE(energy): {rmse_val:.6f}")
    print(f"  NOTE: Bezier energy is a proxy (curvature-based), not true bending energy")
    return {"correlation": corr, "rmse": rmse_val,
            "dz_vals": dz_vals.tolist(), "e_bezier": e_bezier.tolist(), "e_eb": e_eb.tolist()}
