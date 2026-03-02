"""Unit tests for quaternion utilities."""

import math
import numpy as np
import pytest

from morphing_glider.utils.quaternion import (
    quat_normalize, quat_mul, quat_to_rotmat_body_to_world,
    quat_integrate_body_rates, quat_to_euler_xyz,
)


class TestQuatNormalize:
    def test_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        result = quat_normalize(q)
        np.testing.assert_allclose(result, q, atol=1e-10)

    def test_non_unit(self):
        q = np.array([2.0, 0.0, 0.0, 0.0])
        result = quat_normalize(q)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10

    def test_arbitrary(self):
        q = np.array([1.0, 1.0, 1.0, 1.0])
        result = quat_normalize(q)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10


class TestQuatMul:
    def test_identity_mul(self):
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.707, 0.707, 0.0, 0.0])
        result = quat_mul(identity, q)
        np.testing.assert_allclose(result, q, atol=1e-10)

    def test_inverse_mul(self):
        q = quat_normalize(np.array([0.5, 0.5, 0.5, 0.5]))
        q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
        result = quat_mul(q, q_inv)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0], atol=1e-10)


class TestQuatToRotmat:
    def test_identity_rotation(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quat_to_rotmat_body_to_world(q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90deg_roll(self):
        angle = math.pi / 2
        q = np.array([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0])
        R = quat_to_rotmat_body_to_world(q)
        assert R.shape == (3, 3)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10


class TestQuatIntegrate:
    def test_zero_rotation(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 0.0])
        result = quat_integrate_body_rates(q, omega, dt=0.01)
        np.testing.assert_allclose(result, q, atol=1e-10)

    def test_nonzero_rotation(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 1.0])  # yaw rate
        result = quat_integrate_body_rates(q, omega, dt=0.04)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10
        assert result[0] < 1.0  # should deviate from identity


class TestQuatToEuler:
    def test_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        roll, pitch, yaw = quat_to_euler_xyz(q)
        assert abs(roll) < 1e-10
        assert abs(pitch) < 1e-10
        assert abs(yaw) < 1e-10

    def test_pure_roll(self):
        angle = 0.3
        q = np.array([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0])
        roll, pitch, yaw = quat_to_euler_xyz(q)
        assert abs(roll - angle) < 1e-6
        assert abs(pitch) < 1e-6
        assert abs(yaw) < 1e-6
