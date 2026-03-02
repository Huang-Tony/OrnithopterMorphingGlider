"""Unit tests for morphing_glider.environment.observation layout."""

import numpy as np
import pytest

from morphing_glider.environment.observation import OBS_IDX, OBS_DIM


class TestObsDim:
    def test_obs_dim_equals_41(self):
        assert OBS_DIM == 41

    def test_obs_dim_is_int(self):
        assert isinstance(OBS_DIM, int)


class TestObsIdxKeys:
    def test_has_angular_rates(self):
        for key in ("omega_p", "omega_q", "omega_r"):
            assert key in OBS_IDX, f"Missing key: {key}"

    def test_has_yaw_ref(self):
        assert "yaw_ref" in OBS_IDX

    def test_has_speed(self):
        assert "speed" in OBS_IDX

    def test_has_altitude(self):
        assert "altitude" in OBS_IDX

    def test_has_attitude_trig(self):
        for key in ("sin_roll", "cos_roll", "sin_pitch", "cos_pitch",
                     "sin_yaw", "cos_yaw"):
            assert key in OBS_IDX, f"Missing key: {key}"

    def test_has_wing_tip_positions(self):
        for prefix in ("p3_R_", "p3_L_", "p3_cmd_R_", "p3_cmd_L_"):
            for axis in ("x", "y", "z"):
                key = prefix + axis
                assert key in OBS_IDX, f"Missing key: {key}"

    def test_has_bezier_control_points(self):
        for prefix in ("p1_R_", "p2_R_", "p1_L_", "p2_L_"):
            for axis in ("x", "y", "z"):
                key = prefix + axis
                assert key in OBS_IDX, f"Missing key: {key}"


class TestObsIdxValues:
    def test_all_indices_in_range(self):
        for key, idx in OBS_IDX.items():
            assert 0 <= idx < OBS_DIM, (
                f"OBS_IDX['{key}'] = {idx} is out of range [0, {OBS_DIM})"
            )

    def test_all_values_are_ints(self):
        for key, idx in OBS_IDX.items():
            assert isinstance(idx, int), f"OBS_IDX['{key}'] should be int, got {type(idx)}"

    def test_no_duplicate_indices(self):
        values = list(OBS_IDX.values())
        assert len(values) == len(set(values)), "Duplicate indices found in OBS_IDX"

    def test_total_count_matches_dim(self):
        """The number of observation keys should equal OBS_DIM."""
        assert len(OBS_IDX) == OBS_DIM

    def test_indices_are_contiguous(self):
        """Indices should cover 0..OBS_DIM-1 without gaps."""
        assert set(OBS_IDX.values()) == set(range(OBS_DIM))
