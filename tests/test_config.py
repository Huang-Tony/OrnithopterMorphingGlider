"""Unit tests for morphing_glider.config module."""

import numpy as np
import pytest

from morphing_glider.config import (
    DT,
    GLOBAL_SEED,
    RUN_MODE,
    seed_everything,
    L_FIXED,
    WING_CHORD,
    DX_RANGE,
    DY_RANGE,
    DZ_RANGE,
    REWARD_SURVIVAL_BONUS,
    REWARD_TRACKING_SHARPNESS,
)


class TestDT:
    def test_dt_is_positive(self):
        assert DT > 0.0

    def test_dt_reasonable_range(self):
        assert 0.01 <= DT <= 0.1, f"DT={DT} outside expected range [0.01, 0.1]"

    def test_dt_is_float(self):
        assert isinstance(DT, float)


class TestGlobalSeed:
    def test_global_seed_is_int(self):
        assert isinstance(GLOBAL_SEED, int)

    def test_global_seed_nonnegative(self):
        assert GLOBAL_SEED >= 0


class TestRunMode:
    def test_run_mode_valid(self):
        assert RUN_MODE in ("dev", "medium", "paper"), (
            f"RUN_MODE='{RUN_MODE}' not in expected set"
        )

    def test_run_mode_is_string(self):
        assert isinstance(RUN_MODE, str)


class TestSeedEverything:
    def test_seed_everything_no_error(self):
        """seed_everything should complete without raising."""
        seed_everything(42)

    def test_seed_everything_deterministic(self):
        """After seeding, numpy random should produce the same sequence."""
        seed_everything(123)
        a = np.random.rand(5)
        seed_everything(123)
        b = np.random.rand(5)
        np.testing.assert_allclose(a, b)

    def test_seed_everything_different_seeds(self):
        """Different seeds should (almost surely) produce different sequences."""
        seed_everything(0)
        a = np.random.rand(10)
        seed_everything(999)
        b = np.random.rand(10)
        assert not np.allclose(a, b)


class TestGeometryConstants:
    def test_l_fixed_positive(self):
        assert L_FIXED > 0.0

    def test_wing_chord_positive(self):
        assert WING_CHORD > 0.0

    def test_dx_range_is_tuple(self):
        assert isinstance(DX_RANGE, tuple)
        assert len(DX_RANGE) == 2
        assert DX_RANGE[0] < DX_RANGE[1]

    def test_dy_range_is_tuple(self):
        assert isinstance(DY_RANGE, tuple)
        assert len(DY_RANGE) == 2
        assert DY_RANGE[0] < DY_RANGE[1]

    def test_dz_range_is_tuple(self):
        assert isinstance(DZ_RANGE, tuple)
        assert len(DZ_RANGE) == 2
        assert DZ_RANGE[0] < DZ_RANGE[1]


class TestRewardConstants:
    def test_survival_bonus_positive(self):
        assert REWARD_SURVIVAL_BONUS > 0.0

    def test_tracking_sharpness_positive(self):
        assert REWARD_TRACKING_SHARPNESS > 0.0
