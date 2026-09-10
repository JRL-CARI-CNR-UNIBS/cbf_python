"""Unit tests for trajectory generators and trapezoidal interpolation math."""

import numpy as np
import pinocchio as pin
import pytest

from cbf_python.trajectory.trapezoid import trapezoid_coeffs, scalar_trap_unit_progress
from cbf_python.trajectory.joint_interpolator import SegmentedJointTrap
from cbf_python.trajectory.se3_interpolator import SegmentedSE3Trap


def test_trapezoid_coeffs():
    """Verify trapezoid segment durations and distances."""
    v_max = 2.0
    a_max = 4.0
    D = 10.0

    t_acc, t_const, v_peak, total_time = trapezoid_coeffs(D, v_max, a_max)
    assert t_acc > 0
    assert t_const > 0
    assert total_time == pytest.approx(2.0 * t_acc + t_const)
    assert v_peak <= v_max


def test_scalar_trap_unit_progress():
    """Verify unit profile reaches 0 at start and 1 at end."""
    D = 5.0
    v_max = 1.0
    a_max = 1.0
    t_acc, t_const, v_peak, total_time = trapezoid_coeffs(D, v_max, a_max)

    s0, ds0, dds0 = scalar_trap_unit_progress(0.0, t_acc, t_const, v_peak, a_max)
    assert np.isclose(s0, 0.0)

    s_end, ds_end, dds_end = scalar_trap_unit_progress(total_time, t_acc, t_const, v_peak, a_max)
    assert np.isclose(s_end, 1.0)


def test_segmented_joint_trap():
    """Verify multi-joint segmented trajectory."""
    q0 = np.zeros(6)
    q1 = np.ones(6) * 0.5
    q2 = np.ones(6) * -0.2

    planner = SegmentedJointTrap(Dq_max=np.ones(6) * 1.0, DDq_max=np.ones(6) * 2.0)
    planner.addWayPoint(q0)
    planner.addWayPoint(q1)
    planner.addWayPoint(q2)

    total_time = planner.computeTime()
    assert total_time > 0.0

    q_start, _, _ = planner.getMotionLaw(0.0)
    np.testing.assert_allclose(q_start, q0, atol=1e-5)

    q_end, _, _ = planner.getMotionLaw(total_time)
    np.testing.assert_allclose(q_end, q2, atol=1e-5)


def test_segmented_se3_trap():
    """Verify Cartesian SE3 segmented trajectory."""
    T0 = pin.SE3.Identity()
    T1 = pin.SE3(pin.utils.rpyToMatrix(0.1, 0.2, 0.3), np.array([0.5, 0.2, 0.1]))

    planner = SegmentedSE3Trap(vlin_max=0.5, vang_max=0.5, alin_max=1.0, aang_max=1.0)
    planner.addWayPoint(T0)
    planner.addWayPoint(T1)

    total_time = planner.computeTime()
    assert total_time > 0.0

    T_start, _, _ = planner.getMotionLaw(0.0)
    np.testing.assert_allclose(T_start.translation, T0.translation, atol=1e-5)

    T_end, _, _ = planner.getMotionLaw(total_time)
    np.testing.assert_allclose(T_end.translation, T1.translation, atol=1e-5)
