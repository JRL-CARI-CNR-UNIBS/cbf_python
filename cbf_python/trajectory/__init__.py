"""
Trajectory planning and interpolation module.
"""

from cbf_python.trajectory.trapezoid import trapezoid_coeffs, scalar_trap_unit_progress
from cbf_python.trajectory.joint_interpolator import SegmentedJointTrap
from cbf_python.trajectory.se3_interpolator import SegmentedSE3Trap

__all__ = [
    "trapezoid_coeffs",
    "scalar_trap_unit_progress",
    "SegmentedJointTrap",
    "SegmentedSE3Trap",
]
