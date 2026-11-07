import numpy as np

from morphing_glider.config import WIND_X_ABS_MAX_MPS

# Revision R1 (aero model v2): sectional lift/drag calibrated to NASA LTPT wind-tunnel data for the Eppler 387
# at Re = 1e5-2e5 (NASA TM-4062), see revision/calibrate_aero_e387.py and revision/data/aero_calibration_e387.json.
# v1 values (submitted manuscript): cd0=0.03, k_induced=0.06, cl_alpha=5.0, cl_max=1.4, alpha0=0.028,
# alpha_stall=0.25, lift_scale=0.85, tanh saturation, no post-stall lift loss.
NOMINAL_PHYS = {
    "g": 9.81, "rho": 1.225, "V0": 15.0, "mass": 0.5,
    "cd0": 0.0196, "k_induced": 0.0304, "cl_alpha": 5.057, "cl_max": 1.219,
    "cl_sat_n": 26.61, "post_stall_cl_factor": 0.707,
    "alpha0": 0.0235, "alpha_clip": 0.40, "alpha_stall": 0.318, "cd_stall": 0.45,
    "lift_scale": 1.0,
    # Airframe v3 (revision R1): mass properties from a component build-up, pitch damping from thin-airfoil
    # theory, CG at a 10 % static margin, reflex trim (revision/derive_airframe.py -> analysis/airframe_v3.json).
    # Roll/yaw rate damping is resolved by the strip model, so no lumped term is added.
    # v1/v2 lumped values: Ixx = Iyy = 0.25, Izz = 0.12 kg m^2; d_roll = d_pitch = 2.8, d_yaw = 0.25 N m s/rad.
    "Ixx": 0.0534, "Iyy": 0.00421, "Izz": 0.0576,
    "d_roll": 0.0, "d_pitch": 0.0244, "d_yaw": 0.0,
    "x_cg": 0.0128, "cm0": -0.00286,
    "wind_x": 0.0, "wind_y": 0.0, "wind_z": 0.0,
    "gust_tau": 0.8, "gust_sigma_x": 0.5, "gust_sigma_y": 1.0, "gust_sigma_z": 0.6,
    "imu_omega_noise": 0.01, "imu_angle_noise": 0.003, "airspeed_noise": 0.15, "alt_noise": 0.20,
}

class DomainRandomizer:
    def __init__(self, *, enabled=True, scale=1.0):
        self.enabled = bool(enabled); self.scale = float(np.clip(scale, 0.0, 1.0))

    def sample(self, rng):
        base = dict(NOMINAL_PHYS)
        if (not self.enabled) or self.scale <= 0.0: return base
        s = float(self.scale)
        def uni_rel(x, rel):
            rel = float(rel)*s; return float(x*rng.uniform(1.0-rel, 1.0+rel))
        base["rho"]=uni_rel(base["rho"],0.15); base["V0"]=uni_rel(base["V0"],0.20); base["mass"]=uni_rel(base["mass"],0.25)
        base["cd0"]=uni_rel(base["cd0"],0.35); base["k_induced"]=uni_rel(base["k_induced"],0.45)
        base["cl_alpha"]=uni_rel(base["cl_alpha"],0.25); base["cl_max"]=uni_rel(base["cl_max"],0.20)
        base["alpha0"]=uni_rel(base["alpha0"],0.35); base["alpha_stall"]=uni_rel(base["alpha_stall"],0.20)
        base["cd_stall"]=uni_rel(base["cd_stall"],0.45); base["lift_scale"]=uni_rel(base["lift_scale"],0.30)
        base["Ixx"]=uni_rel(base["Ixx"],0.35); base["Iyy"]=uni_rel(base["Iyy"],0.35); base["Izz"]=uni_rel(base["Izz"],0.35)
        base["d_roll"]=uni_rel(base["d_roll"],0.50); base["d_pitch"]=uni_rel(base["d_pitch"],0.50); base["d_yaw"]=uni_rel(base["d_yaw"],0.50)
        base["wind_x"]=float(rng.uniform(-WIND_X_ABS_MAX_MPS*s, +WIND_X_ABS_MAX_MPS*s))
        base["wind_y"]=float(rng.uniform(-2.5*s, +2.5*s)); base["wind_z"]=float(rng.uniform(-1.5*s, +1.5*s))
        base["gust_tau"]=float(np.clip(uni_rel(base["gust_tau"],0.60),0.15,3.0))
        base["gust_sigma_x"]=float(np.clip(uni_rel(base["gust_sigma_x"],1.0),0.0,3.0))
        base["gust_sigma_y"]=float(np.clip(uni_rel(base["gust_sigma_y"],1.0),0.0,4.0))
        base["gust_sigma_z"]=float(np.clip(uni_rel(base["gust_sigma_z"],1.0),0.0,3.0))
        base["imu_omega_noise"]=float(base["imu_omega_noise"]*(1.0+1.2*s))
        base["imu_angle_noise"]=float(base["imu_angle_noise"]*(1.0+1.2*s))
        base["airspeed_noise"]=float(base["airspeed_noise"]*(1.0+1.0*s))
        base["alt_noise"]=float(base["alt_noise"]*(1.0+1.0*s))
        base["x_cg"]=uni_rel(base["x_cg"],0.30)     # CG uncertainty: static margin ~7-13 %
        return base
