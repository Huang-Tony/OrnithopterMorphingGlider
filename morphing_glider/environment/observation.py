"""Observation space layout."""

OBS_IDX = {
    "sin_roll": 0, "cos_roll": 1, "sin_pitch": 2, "cos_pitch": 3,
    "sin_yaw": 4, "cos_yaw": 5,
    "omega_p": 6, "omega_q": 7, "omega_r": 8,
    "v_rel_u": 9, "v_rel_v": 10, "v_rel_w": 11, "speed": 12,
    "altitude": 13, "vz_world": 14,
    "yaw_ref": 15, "yaw_ref_prev": 16,
    "p3_R_x": 17, "p3_R_y": 18, "p3_R_z": 19,
    "p3_L_x": 20, "p3_L_y": 21, "p3_L_z": 22,
    "p3_cmd_R_x": 23, "p3_cmd_R_y": 24, "p3_cmd_R_z": 25,
    "p3_cmd_L_x": 26, "p3_cmd_L_y": 27, "p3_cmd_L_z": 28,
    "p1_R_x": 29, "p1_R_y": 30, "p1_R_z": 31,
    "p2_R_x": 32, "p2_R_y": 33, "p2_R_z": 34,
    "p1_L_x": 35, "p1_L_y": 36, "p1_L_z": 37,
    "p2_L_x": 38, "p2_L_y": 39, "p2_L_z": 40,
}
OBS_DIM = 41
