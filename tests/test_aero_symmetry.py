"""Regression tests for the revision-R1 camber-sign fix in AeroProxy3D (mirror symmetry of the wing pair)."""

import math

import numpy as np
import pytest

from morphing_glider.config import L_FIXED
from morphing_glider.physics.aero_proxy import AeroProxy3D
from morphing_glider.physics.bezier_spar import RealTimeBezierSpar
from morphing_glider.physics.domain_randomizer import NOMINAL_PHYS


def _spar(p3):
    s = np.sign(p3[1])
    sp = RealTimeBezierSpar([0, 0, 0], p3, [0, s * 0.33 * L_FIXED, 0], [0, s * 0.66 * L_FIXED, 0])
    sp.iterations = 12
    sp.solve_to_convergence(max_total_iters=96, chunk_iters=12)
    return sp


def _loads(sR, sL, v, omega=np.zeros(3)):
    aero = AeroProxy3D(num_panels=10)
    FR, MR, _ = aero.calculate_forces(sR, v_rel_body=v, omega_body=omega, phys=dict(NOMINAL_PHYS))
    FL, ML, _ = aero.calculate_forces(sL, v_rel_body=v, omega_body=omega, phys=dict(NOMINAL_PHYS))
    return FR, MR, FL, ML


@pytest.mark.parametrize("alpha_deg", [0.0, 2.0, 5.0])
def test_symmetric_wings_produce_no_roll_or_yaw(alpha_deg):
    a = math.radians(alpha_deg)
    v = 15.0 * np.array([math.cos(a), 0.0, -math.sin(a)])
    FR, MR, FL, ML = _loads(_spar([0, L_FIXED, 0]), _spar([0, -L_FIXED, 0]), v)
    assert abs(FR[2] - FL[2]) < 1e-6                    # equal vertical force on both wings
    assert abs((MR + ML)[0]) < 1e-6                      # no rolling moment
    assert abs((MR + ML)[2]) < 1e-6                      # no yawing moment
    assert (FR + FL)[2] > 0.0                            # positive net lift (camber offset alpha0 > 0)


def test_level_trim_lift_close_to_weight():
    FR, _, FL, _ = _loads(_spar([0, L_FIXED, 0]), _spar([0, -L_FIXED, 0]), np.array([15.0, 0.0, 0.0]))
    weight = NOMINAL_PHYS["mass"] * NOMINAL_PHYS["g"]
    assert 0.7 * weight < (FR + FL)[2] < 1.3 * weight


@pytest.mark.parametrize("dx,dz", [(0.2, 0.0), (0.3, 0.1), (-0.25, 0.05)])
def test_mirrored_deflection_gives_mirrored_moments(dx, dz):
    v = np.array([15.0, 0.0, 0.0])
    # A: right (+y) wing deflected, left neutral.  B: mirror image.
    _, MRa, _, MLa = _loads(_spar([dx, L_FIXED, dz]), _spar([0, -L_FIXED, 0]), v)
    _, MRb, _, MLb = _loads(_spar([0, L_FIXED, 0]), _spar([dx, -L_FIXED, dz]), v)
    Ma, Mb = MRa + MLa, MRb + MLb
    assert Ma[0] == pytest.approx(-Mb[0], abs=1e-6)     # roll flips sign
    assert Ma[1] == pytest.approx(Mb[1], abs=1e-6)      # pitch unchanged
    assert Ma[2] == pytest.approx(-Mb[2], abs=1e-6)     # yaw flips sign


def test_differential_sweep_produces_yaw():
    v = np.array([15.0, 0.0, 0.0])
    _, M0R, _, M0L = _loads(_spar([0, L_FIXED, 0]), _spar([0, -L_FIXED, 0]), v)
    _, MR, _, ML = _loads(_spar([0.2, L_FIXED, 0]), _spar([-0.2, -L_FIXED, 0]), v)
    assert abs((MR + ML - M0R - M0L)[2]) > 0.02          # N*m: differential sweep is a yaw effector
