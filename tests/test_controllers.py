"""Unit tests for morphing glider controllers."""

import numpy as np
import pytest

from morphing_glider.controllers.zero import ZeroController
from morphing_glider.controllers.pid import PIDYawController
from morphing_glider.controllers.lqr import LQRYawController
from morphing_glider.controllers.heuristic import VirtualTendonHeuristicController
from morphing_glider.environment.observation import OBS_DIM


def _make_dummy_obs(yaw_ref=0.3, omega_r=0.0, speed=15.0):
    """Build a minimal valid observation vector."""
    from morphing_glider.environment.observation import OBS_IDX
    obs = np.zeros(OBS_DIM, dtype=np.float64)
    obs[OBS_IDX["omega_r"]] = omega_r
    obs[OBS_IDX["yaw_ref"]] = yaw_ref
    obs[OBS_IDX["speed"]] = speed
    # Give a unit quaternion for cos_roll, cos_pitch
    obs[OBS_IDX["cos_roll"]] = 1.0
    obs[OBS_IDX["cos_pitch"]] = 1.0
    return obs


# ----------------------------------------------------------------
# ZeroController
# ----------------------------------------------------------------
class TestZeroController:
    def test_instantiation(self):
        ctrl = ZeroController()
        assert ctrl is not None

    def test_predict_returns_zeros(self):
        ctrl = ZeroController()
        obs = np.zeros(OBS_DIM)
        action, state = ctrl.predict(obs)
        np.testing.assert_allclose(action, np.zeros(6), atol=1e-10)

    def test_predict_action_shape(self):
        ctrl = ZeroController()
        action, _ = ctrl.predict(np.zeros(OBS_DIM))
        assert action.shape == (6,)

    def test_predict_state_passthrough(self):
        ctrl = ZeroController()
        sentinel = object()
        _, state = ctrl.predict(np.zeros(OBS_DIM), state=sentinel)
        assert state is sentinel

    def test_reset_does_not_error(self):
        ctrl = ZeroController()
        ctrl.reset()  # should not raise


# ----------------------------------------------------------------
# PIDYawController
# ----------------------------------------------------------------
class TestPIDYawController:
    def test_instantiation(self):
        ctrl = PIDYawController()
        assert ctrl is not None

    def test_predict_shape(self):
        ctrl = PIDYawController()
        obs = _make_dummy_obs(yaw_ref=0.3, omega_r=0.0)
        action, state = ctrl.predict(obs)
        assert action.shape == (6,)

    def test_predict_dtype(self):
        ctrl = PIDYawController()
        action, _ = ctrl.predict(_make_dummy_obs())
        assert action.dtype == np.float32

    def test_nonzero_output_for_error(self):
        ctrl = PIDYawController()
        obs = _make_dummy_obs(yaw_ref=0.5, omega_r=0.0)
        action, _ = ctrl.predict(obs)
        # With a positive yaw error, the controller should produce nonzero action
        assert np.any(np.abs(action) > 1e-6)

    def test_antisymmetric_action(self):
        """PID action has opposite dx on right vs left wing."""
        ctrl = PIDYawController()
        obs = _make_dummy_obs(yaw_ref=0.3, omega_r=0.0)
        action, _ = ctrl.predict(obs)
        # action[0] is right dx, action[3] is left dx; they should be negatives
        np.testing.assert_allclose(action[0], -action[3], atol=1e-10)

    def test_reset_clears_state(self):
        ctrl = PIDYawController()
        obs = _make_dummy_obs(yaw_ref=0.5, omega_r=0.0)
        ctrl.predict(obs)
        ctrl.predict(obs)
        assert ctrl._integral != 0.0 or ctrl._prev_error != 0.0
        ctrl.reset()
        assert ctrl._integral == 0.0
        assert ctrl._prev_error == 0.0

    def test_zero_error_zero_output(self):
        """When yaw_ref equals omega_r, first call should produce zero output."""
        ctrl = PIDYawController()
        obs = _make_dummy_obs(yaw_ref=0.0, omega_r=0.0)
        action, _ = ctrl.predict(obs)
        np.testing.assert_allclose(action, np.zeros(6), atol=1e-10)


# ----------------------------------------------------------------
# LQRYawController
# ----------------------------------------------------------------
class TestLQRYawController:
    def test_instantiation(self):
        ctrl = LQRYawController()
        assert ctrl is not None

    def test_predict_shape(self):
        ctrl = LQRYawController()
        obs = _make_dummy_obs(yaw_ref=0.3, omega_r=0.0)
        action, state = ctrl.predict(obs)
        assert action.shape == (6,)

    def test_predict_dtype(self):
        ctrl = LQRYawController()
        action, _ = ctrl.predict(_make_dummy_obs())
        assert action.dtype == np.float32

    def test_nonzero_output_for_error(self):
        ctrl = LQRYawController()
        obs = _make_dummy_obs(yaw_ref=0.5, omega_r=0.0)
        action, _ = ctrl.predict(obs)
        assert np.any(np.abs(action) > 1e-6)

    def test_antisymmetric_action(self):
        ctrl = LQRYawController()
        obs = _make_dummy_obs(yaw_ref=0.3, omega_r=0.0)
        action, _ = ctrl.predict(obs)
        np.testing.assert_allclose(action[0], -action[3], atol=1e-10)

    def test_gain_is_positive(self):
        ctrl = LQRYawController()
        assert ctrl.K > 0.0


# ----------------------------------------------------------------
# VirtualTendonHeuristicController
# ----------------------------------------------------------------
class TestVirtualTendonHeuristicController:
    def test_instantiation(self):
        ctrl = VirtualTendonHeuristicController()
        assert ctrl is not None

    def test_predict_shape(self):
        ctrl = VirtualTendonHeuristicController()
        obs = _make_dummy_obs(yaw_ref=0.3, omega_r=0.0)
        action, state = ctrl.predict(obs)
        assert action.shape == (6,)

    def test_predict_dtype(self):
        ctrl = VirtualTendonHeuristicController()
        action, _ = ctrl.predict(_make_dummy_obs())
        assert action.dtype == np.float32

    def test_nonzero_output_for_error(self):
        ctrl = VirtualTendonHeuristicController()
        obs = _make_dummy_obs(yaw_ref=0.5, omega_r=0.0)
        action, _ = ctrl.predict(obs)
        assert np.any(np.abs(action) > 1e-6)

    def test_reset_clears_previous_command(self):
        ctrl = VirtualTendonHeuristicController()
        obs = _make_dummy_obs(yaw_ref=0.5, omega_r=0.0)
        ctrl.predict(obs)
        ctrl.reset()
        np.testing.assert_allclose(ctrl._prev_cmd, np.zeros(6), atol=1e-10)

    def test_zero_error_within_deadband(self):
        """When error is within deadband, output should be small."""
        ctrl = VirtualTendonHeuristicController(deadband=0.1)
        obs = _make_dummy_obs(yaw_ref=0.05, omega_r=0.0)
        action, _ = ctrl.predict(obs)
        # With smoothing from zero prev_cmd and deadband filtering, action should be zero
        np.testing.assert_allclose(action, np.zeros(6), atol=1e-10)
