import numpy as np
import matplotlib.pyplot as plt

from morphing_glider.config import (
    FAST_DEV_RUN, L_FIXED, WING_CHORD, EVAL_AERO_PANELS, BEZIER_ITERS_EVAL,
    REWARD_W_STRUCT, REWARD_W_TRACK, _save_fig, _add_panel_label,
)
from morphing_glider.physics import RealTimeBezierSpar, AeroProxy3D, NOMINAL_PHYS
from morphing_glider.environment.env import MorphingGliderEnv6DOF


def aero_calibration():
    phys = dict(NOMINAL_PHYS); phys["wind_x"]=0.0; phys["wind_y"]=0.0; phys["wind_z"]=0.0
    aero = AeroProxy3D(num_panels=EVAL_AERO_PANELS, include_omega_cross=True)
    v_body = np.array([float(phys["V0"]),0.0,0.0])
    spar_L = RealTimeBezierSpar([0,0,0],[0,-L_FIXED,0],[0,-L_FIXED*0.33,0],[0,-L_FIXED*0.66,0])
    spar_L.iterations = BEZIER_ITERS_EVAL; spar_L.solve_to_convergence()
    dx=0.20; dz=0.10
    spar_R = RealTimeBezierSpar([0,0,0],[dx,+L_FIXED,dz],[0,+L_FIXED*0.33,0],[0,+L_FIXED*0.66,0])
    spar_R.iterations = BEZIER_ITERS_EVAL; spar_R.solve_to_convergence()
    F_R, M_R, d_R = aero.calculate_forces(spar_R, v_rel_body=v_body, omega_body=np.zeros(3), phys=phys)
    F_L, M_L, d_L = aero.calculate_forces(spar_L, v_rel_body=v_body, omega_body=np.zeros(3), phys=phys)
    Mz_total = float((M_R+M_L)[2]); Izz = float(phys["Izz"])
    out = {"dx_ref": dx, "dz_ref": dz, "V0": float(phys["V0"]), "cl_alpha": float(phys["cl_alpha"]),
           "lift_scale": float(phys["lift_scale"]), "Izz": Izz, "Mz_total": Mz_total,
           "yaw_acc": Mz_total/max(1e-9, Izz),
           "drag_R": float(d_R["total_drag_force"]), "drag_L": float(d_L["total_drag_force"])}
    print("\n" + "="*80)
    print("[AERO CALIBRATION]")
    print(f"  Mz_total={Mz_total:+.4f} N·m | yaw_acc={out['yaw_acc']:+.3f} rad/s²")
    print(f"  cl_alpha={phys['cl_alpha']:.3f} | lift_scale={phys['lift_scale']:.3f}")

    # Trim check
    S = WING_CHORD*L_FIXED*2; ls = float(phys["lift_scale"]); V0 = float(phys["V0"])
    F_lift = 0.5*1.225*V0**2*S*float(phys["cl_alpha"])*float(phys.get("alpha0",0.0))*ls
    weight = float(phys["mass"])*9.81
    print(f"  Trim: lift_est={F_lift:.3f} N, weight={weight:.3f} N, ratio={F_lift/max(weight,1e-6):.3f}")

    # Reward check
    try:
        _, eR = spar_R.length_and_energy(); _, eL = spar_L.length_and_energy()
        e_max = MorphingGliderEnv6DOF.get_e_sum_max_cached()
        cs = float(np.clip((eR+eL)/max(e_max,1e-3), 0.0, 2.0))
        ws = REWARD_W_STRUCT*cs; wt = 0.20*REWARD_W_TRACK*0.36
        print(f"  Reward check: w_struct*cost={ws:.6f} vs 20%*w_track*cost={wt:.6f} {'OK' if ws<wt else 'WARN'}")
    except Exception: pass
    return out


def aero_sanity_sweep():
    phys = dict(NOMINAL_PHYS); phys["wind_x"]=0.0; phys["wind_y"]=0.0; phys["wind_z"]=0.0
    aero = AeroProxy3D(num_panels=EVAL_AERO_PANELS, include_omega_cross=True)
    v_body = np.array([float(phys["V0"]),0.0,0.0])
    spar_R0 = RealTimeBezierSpar([0,0,0],[0,+L_FIXED,0],[0,+L_FIXED*0.33,0],[0,+L_FIXED*0.66,0])
    spar_L0 = RealTimeBezierSpar([0,0,0],[0,-L_FIXED,0],[0,-L_FIXED*0.33,0],[0,-L_FIXED*0.66,0])
    spar_R0.iterations = BEZIER_ITERS_EVAL; spar_L0.iterations = BEZIER_ITERS_EVAL
    spar_R0.solve_to_convergence(); spar_L0.solve_to_convergence()
    _, M_R0, _ = aero.calculate_forces(spar_R0, v_rel_body=v_body, omega_body=np.zeros(3), phys=phys)
    _, M_L0, _ = aero.calculate_forces(spar_L0, v_rel_body=v_body, omega_body=np.zeros(3), phys=phys)
    Mz_base = float(M_R0[2]+M_L0[2])
    n = 14 if FAST_DEV_RUN else 28
    xs = np.linspace(0.0, 0.45, n); zs = np.linspace(0.0, 0.15, n)
    Mz_grid = np.zeros((zs.size, xs.size))
    for i, z in enumerate(zs):
        for j, x in enumerate(xs):
            if i==0 and j==0: continue
            spar_R0.p3 = np.array([float(x), +L_FIXED, float(z)])
            spar_R0.solve_to_convergence(max_total_iters=80, chunk_iters=12, tol_len=1e-3)
            _, M_R_def, _ = aero.calculate_forces(spar_R0, v_rel_body=v_body, omega_body=np.zeros(3), phys=phys)
            Mz_grid[i, j] = float(M_R_def[2]+M_L0[2]) - Mz_base
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(Mz_grid, origin="lower", aspect="auto", extent=[xs.min(),xs.max(),zs.min(),zs.max()])
    plt.colorbar(im, ax=ax, label="ΔMz (N·m)"); ax.set_xlabel("dx (m)"); ax.set_ylabel("dz (m)")
    ax.set_title("Yaw authority ΔMz"); _add_panel_label(ax, "A")
    plt.tight_layout(); _save_fig(fig, "aero_sanity.png", "Yaw authority heatmap")
    plt.show()
