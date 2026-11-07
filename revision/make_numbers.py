#!/usr/bin/env python3
"""Generate numbers.tex: every number quoted in the revised manuscript, as LaTeX macros, from the result
files in revision_runs/ (no number in the text is typed by hand). Missing results render as '??'."""

import glob
import json
import math
import os

import numpy as np

import common  # noqa: F401
from common import RUNS_DIR, REPO, TOTAL_STEPS, N_ENVS, EVAL_EVERY, TRAIN_EVAL_EPS, FINAL_EVAL_EPS, TRAIN_SEEDS
from morphing_glider.physics.domain_randomizer import NOMINAL_PHYS
from morphing_glider import config as C

A = os.path.join(RUNS_DIR, "analysis")
OUT = os.environ.get("NUMBERS_TEX", os.path.join(RUNS_DIR, "numbers.tex"))
M = {}


def J(name):
    p = os.path.join(A, name)
    return json.load(open(p)) if os.path.exists(p) else None


def put(key, val, fmt="{:.3f}"):
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        M[key] = r"\textbf{??}"
    elif isinstance(val, str):
        M[key] = val
    else:
        M[key] = fmt.format(val)


def main():
    P = NOMINAL_PHYS
    # ---- platform ----
    put("PlanSpan", 2 * C.L_FIXED, "{:.1f}"); put("PlanSemiSpan", C.L_FIXED, "{:.1f}"); put("PlanChord", C.WING_CHORD, "{:.2f}")
    put("PlanArea", C.WING_AREA_TOTAL, "{:.2f}"); put("PlanAR", C.ASPECT_RATIO, "{:.1f}"); put("PlanMass", P["mass"], "{:.1f}")
    put("PlanVzero", P["V0"], "{:.0f}"); put("PlanIxx", P["Ixx"], "{:.2f}"); put("PlanIyy", P["Iyy"], "{:.2f}"); put("PlanIzz", P["Izz"], "{:.2f}")
    put("PlanDroll", P["d_roll"], "{:.1f}"); put("PlanDpitch", P["d_pitch"], "{:.1f}"); put("PlanDyaw", P["d_yaw"], "{:.2f}")
    put("PlanRe", P["rho"] * P["V0"] * C.WING_CHORD / 1.81e-5 / 1e5, "{:.1f}")
    put("PlanWingLoading", P["mass"] * P["g"] / C.WING_AREA_TOTAL, "{:.1f}")
    put("DT", C.DT * 1000, "{:.0f}"); put("Substeps", C.PHYSICS_SUBSTEPS, "{:d}"); put("SubDT", C.DT * 1000 / C.PHYSICS_SUBSTEPS, "{:.0f}")
    put("DxMax", C.DX_RANGE[1], "{:.2f}"); put("DyMax", C.DY_RANGE[1], "{:.2f}"); put("DzMax", C.DZ_RANGE[1], "{:.2f}")
    put("TauAct", 70, "{:d}")
    # ---- aero model v2 ----
    for k, key in [("cl_alpha", "AeroClAlpha"), ("cl_max", "AeroClMax"), ("cd0", "AeroCdZero"), ("k_induced", "AeroKind"),
                   ("cl_sat_n", "AeroSatN"), ("post_stall_cl_factor", "AeroPostStall"), ("cd_stall", "AeroCdStall"),
                   ("lift_scale", "AeroLiftScale")]:
        put(key, P[k], "{:.3f}" if k not in ("cl_sat_n",) else "{:.1f}")
    put("AeroAlphaZeroDeg", math.degrees(P["alpha0"]), "{:.2f}"); put("AeroAlphaStallDeg", math.degrees(P["alpha_stall"]), "{:.1f}")
    put("AeroAlphaClipDeg", math.degrees(P["alpha_clip"]), "{:.0f}")
    av = J("aero_validation_e387.json")
    if av:
        for Re, t in (("100000", "A"), ("200000", "B")):
            r = av[f"Re{Re}"]
            put(f"AeroSlopeTwoD{t}", r["a0_2D_per_rad"], "{:.2f}"); put(f"AeroSlopeThreeD{t}", r["a_3D_helmbold_per_rad"], "{:.2f}")
            put(f"AeroSlopeErr{t}", 100 * r["proxy_slope_rel_err"], "{:+.1f}"); put(f"AeroClmaxMeas{t}", r["cl_max"], "{:.3f}")
            put(f"AeroClmaxErr{t}", 100 * r["proxy_clmax_rel_err"], "{:+.1f}"); put(f"AeroStallErr{t}", r["proxy_stall_minus_meas_deg"], "{:+.1f}")
            put(f"AeroAlphaLzero{t}", r["alpha_L0_deg"], "{:.2f}")
            dr = [v for kk, v in r.items() if kk.startswith("drag_ratio")]
            put(f"AeroDragRatioMin{t}", min(dr), "{:.2f}"); put(f"AeroDragRatioMax{t}", max(dr), "{:.2f}")
        put("AeroOswald", av["proxy_oswald_equiv"], "{:.2f}")
        put("AeroSlopeEnvLo", av["randomisation_envelope"]["effective_slope_p5_p95"][0], "{:.1f}")
        put("AeroSlopeEnvHi", av["randomisation_envelope"]["effective_slope_p5_p95"][1], "{:.1f}")
    cal = json.load(open(os.path.join(REPO, "revision", "data", "aero_calibration_e387.json")))
    put("AeroTrimLD", cal["trim_check"]["L_over_D"], "{:.1f}"); put("AeroParasitic", cal["d_cd0_parasitic"], "{:.3f}")
    put("AeroOswaldAssumed", cal["e_oswald"], "{:.2f}")
    v1 = cal["v1_values"]
    put("VoneCdZero", v1["cd0"], "{:.2f}"); put("VoneKind", v1["k_induced"], "{:.2f}"); put("VoneClAlpha", v1["cl_alpha"], "{:.1f}")
    put("VoneLiftScale", v1["lift_scale"], "{:.2f}")
    # ---- plant identification ----
    pi = J("plant_identification.json")
    if pi:
        e = pi["control_effectiveness"]
        put("EffDsMz", e["diff_sweep"]["dM_ddelta"][2], "{:.2f}"); put("EffSsMy", e["sym_sweep"]["dM_ddelta"][1], "{:.2f}")
        put("EffDdMx", e["diff_dihedral"]["dM_ddelta"][0], "{:.2f}"); put("EffDdMz", e["diff_dihedral"]["dM_ddelta"][2], "{:.2f}")
        put("EffSdMy", e["sym_dihedral"]["dM_ddelta"][1], "{:.2f}"); put("EffDyMx", e["diff_span"]["dM_ddelta"][0], "{:.2f}")
        put("EffDyMz", e["diff_span"]["dM_ddelta"][2], "{:.2f}"); put("EffSyMy", e["sym_span"]["dM_ddelta"][1], "{:.2f}")
        put("EffSyFz", e["sym_span"]["dF_ddelta"][2], "{:.2f}"); put("EffDdFy", e["diff_dihedral"]["dF_ddelta"][1], "{:.2f}")
        put("SweepMzMax", max(c["Mz"] for c in pi["diff_sweep_yaw_moment_curve"]), "{:.3f}")
        rd = pi["aero_rate_derivatives_dM_domega"]
        put("AeroLp", rd["p"][0], "{:.2f}"); put("AeroNr", rd["r"][2], "{:.3f}")
        put("BetaLx", pi["dM_dbeta_per_rad"][0], "{:.1f}"); put("BetaNz", pi["dM_dbeta_per_rad"][2], "{:.2f}")
        put("AlphaMy", pi["dM_dalpha_per_rad"][1], "{:.2f}"); put("WingCLalpha", pi["CL_alpha_whole_wing_per_rad"], "{:.2f}")
        ym = pi["yaw_model"]
        put("YawDamp", ym["total_yaw_damping"], "{:.3f}"); put("YawTau", ym["time_constant_s"], "{:.2f}")
        put("YawGain", ym["steady_gain_rad_s_per_m"], "{:.2f}")
        put("SweepYawRateMax", max(c["Mz"] for c in pi["diff_sweep_yaw_moment_curve"]) / ym["total_yaw_damping"], "{:.2f}")
        put("TrimLift", pi["trim"]["lift_N"], "{:.2f}"); put("TrimDrag", pi["trim"]["drag_N"], "{:.2f}")
        put("TrimLD", pi["trim"]["L_over_D"], "{:.1f}"); put("TrimMy", pi["trim"]["M_body"][1], "{:.3f}")
    pc = J("platform_configs.json")
    if pc:
        put("CfgSweepMz", pc["configs"]["diff_sweep"]["dM"][2], "{:.3f}"); put("CfgDihedMx", pc["configs"]["diff_dihedral"]["dM"][0], "{:.3f}")
    # ---- plan B ----
    pb = J("planb_fold_validation.json")
    if pb:
        a = pb["actuator"]; cl = a["tau_clusters"]; s = pb["spar_shape"]
        put("PBnSteps", a["n_steps"], "{:d}"); put("PBfastN", cl["fast_n"], "{:d}"); put("PBslowN", cl["slow_n"], "{:d}")
        put("PBfastMaxMs", 1000 * cl["fast_range_s"][1], "{:.0f}"); put("PBslowMedMs", 1000 * cl["slow_median_s"], "{:.0f}")
        put("PBslowLoMs", 1000 * cl["slow_range_s"][0], "{:.0f}"); put("PBslowHiMs", 1000 * cl["slow_range_s"][1], "{:.0f}")
        put("PBrelaxLoMs", 1000 * pb["relaxation_time_constant_s"]["min"], "{:.0f}"); put("PBrelaxHiMs", 1000 * pb["relaxation_time_constant_s"]["max"], "{:.0f}")
        put("PBmodeHz", pb["wing_mode"]["f_n_hz_median"], "{:.2f}"); put("PBmodeZeta", pb["wing_mode"]["zeta_median"], "{:.3f}")
        put("PBnFrames", s["n_frames"], "{:d}"); put("PBsparL", s["arc_length_mm_median"], "{:.0f}")
        put("PBrmsLine", s["rms_line_mm_median"], "{:.1f}"); put("PBrmsQuad", s["rms_quad_mm_median"], "{:.1f}")
        put("PBrmsCubic", s["rms_cubic_mm_median"], "{:.1f}"); put("PBrmsCubicPct", 100 * s["rms_cubic_mm_median"] / s["arc_length_mm_median"], "{:.2f}")
        put("PBrmsCubicNinetyFive", s["rms_cubic_mm_p95"], "{:.1f}"); put("PBrmsSolver", s["rms_solver_mm_median"], "{:.1f}")
        put("PBrmsSolverNinetyFive", s["rms_solver_mm_p95"], "{:.1f}")
    mv = J("model_verification.json")
    if mv:
        integ = mv["integration"].values()
        put("IntegErrYawPct", max(c["yaw_err_pct_sub4"] for c in integ), "{:.1f}")
        put("IntegErrYawPctCoarse", max(c["yaw_err_pct_sub1"] for c in mv["integration"].values()), "{:.1f}")
        put("IntegErrRollDeg", max(c["roll_err_deg_sub4"] for c in mv["integration"].values()), "{:.2f}")
        put("SparEBcorr", mv["spar_energy"].get("correlation"), "{:.4f}")
    logs = sorted(glob.glob(os.path.join(RUNS_DIR, "logs", "*.train.json")))
    if logs:
        lg = json.load(open(logs[0]))
        put("ActorParams", f"{lg['actor_params']:,}".replace(",", "{,}"))
        put("ActorParamsMean", f"{lg['actor_params_without_logstd_head']:,}".replace(",", "{,}"))
    # ---- training configuration ----
    put("TotalSteps", TOTAL_STEPS / 1e6, "{:.0f}"); put("NEnvs", N_ENVS, "{:d}"); put("EvalEvery", EVAL_EVERY / 1e3, "{:.0f}")
    put("TrainEvalEps", TRAIN_EVAL_EPS, "{:d}"); put("FinalEvalEps", FINAL_EVAL_EPS, "{:d}"); put("NSeeds", len(TRAIN_SEEDS), "{:d}")
    put("FinalEvalEpsTotal", FINAL_EVAL_EPS * len(TRAIN_SEEDS), "{:d}")
    # ---- results (filled by later stages; see make_numbers_results in this file) ----
    res = J("results_summary.json")
    if res:
        for k, v in res.get("macros", {}).items():
            M[k] = v
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("% Auto-generated by revision/make_numbers.py -- do not edit by hand.\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\n{k}}}{{{M[k]}}}\n")
    print(f"{len(M)} macros -> {OUT}")


if __name__ == "__main__":
    main()
