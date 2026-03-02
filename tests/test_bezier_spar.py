"""Unit tests for morphing_glider.physics.bezier_spar.RealTimeBezierSpar."""

import numpy as np
import pytest

from morphing_glider.physics.bezier_spar import RealTimeBezierSpar


@pytest.fixture
def spar():
    """Create a default spar for testing."""
    p0 = np.array([0.0, 0.0, 0.0])
    p3 = np.array([1.0, 0.0, 0.0])
    p1 = np.array([0.33, 0.0, 0.05])
    p2 = np.array([0.66, 0.0, 0.05])
    return RealTimeBezierSpar(p0, p3, p1, p2)


class TestRealTimeBezierSparInstantiation:
    def test_creates_spar(self, spar):
        assert isinstance(spar, RealTimeBezierSpar)

    def test_stores_control_points(self, spar):
        np.testing.assert_allclose(spar.p0, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(spar.p3, [1.0, 0.0, 0.0])

    def test_has_default_parameters(self, spar):
        assert spar.learning_rate == 0.04
        assert spar.iterations == 10
        assert spar.lock_z is False


class TestEvaluate:
    def test_single_point_shape(self, spar):
        """evaluate() at a single parameter returns shape (3,)."""
        result = spar.evaluate(0.5)
        assert result.shape == (3,)

    def test_multiple_points_shape(self, spar):
        """evaluate() on an array returns (n_pts, 3) array."""
        u = np.linspace(0, 1, 20)
        result = spar.evaluate(u.reshape(-1, 1))
        assert result.shape == (20, 3)

    def test_endpoints(self, spar):
        """Bezier curve passes through p0 at u=0 and p3 at u=1."""
        start = spar.evaluate(0.0)
        end = spar.evaluate(1.0)
        np.testing.assert_allclose(start, spar.p0, atol=1e-10)
        np.testing.assert_allclose(end, spar.p3, atol=1e-10)

    def test_midpoint_between_endpoints(self, spar):
        """Midpoint should lie between the endpoints (bounding box property)."""
        mid = spar.evaluate(0.5)
        assert 0.0 <= mid[0] <= 1.0


class TestTangent:
    def test_tangent_shape(self, spar):
        """tangent() at a scalar returns shape (3,)."""
        result = spar.tangent(0.5)
        assert result.shape == (3,)

    def test_tangent_at_start(self, spar):
        """Tangent at u=0 is proportional to 3*(p1 - p0)."""
        result = spar.tangent(0.0)
        expected = 3.0 * (spar.p1 - spar.p0)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_tangent_at_end(self, spar):
        """Tangent at u=1 is proportional to 3*(p3 - p2)."""
        result = spar.tangent(1.0)
        expected = 3.0 * (spar.p3 - spar.p2)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_tangent_nonzero(self, spar):
        """Tangent at an interior point should have nonzero magnitude."""
        result = spar.tangent(0.5)
        assert np.linalg.norm(result) > 1e-6


class TestDeformedVsUndeformed:
    def test_deformed_shape_differs(self):
        """Moving p3 should change the evaluated curve."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.33, 0.0, 0.0])
        p2 = np.array([0.66, 0.0, 0.0])

        spar_straight = RealTimeBezierSpar(p0, np.array([1.0, 0.0, 0.0]), p1, p2)
        spar_bent = RealTimeBezierSpar(p0, np.array([1.0, 0.0, 0.2]), p1, p2)

        u = np.array([0.5]).reshape(-1, 1)
        pts_straight = spar_straight.evaluate(u)
        pts_bent = spar_bent.evaluate(u)

        # The z-component at midpoint should differ
        assert not np.allclose(pts_straight, pts_bent)

    def test_solve_shape_changes_control_points(self, spar):
        """solve_shape() should modify p1 and p2."""
        p1_before = spar.p1.copy()
        p2_before = spar.p2.copy()
        spar.solve_shape(iterations=5)
        changed = (not np.allclose(spar.p1, p1_before) or
                   not np.allclose(spar.p2, p2_before))
        assert changed, "solve_shape should modify at least one control point"


class TestLengthAndEnergy:
    def test_length_positive(self, spar):
        length, energy = spar.length_and_energy()
        assert length > 0.0

    def test_energy_nonnegative(self, spar):
        length, energy = spar.length_and_energy()
        assert energy >= 0.0
