"""
Trapezoidal motion profile mathematics.

This module provides 1-D trapezoidal velocity profile calculation
and normalized unit progression scaling for joint-space and Cartesian SE3 interpolators.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional


def trapezoid_coeffs(dist: float, vmax: float, amax: float) -> Tuple[float, float, float, float]:
    """Compute (t_acc, t_const, v_peak, total_time) for a 1-D trapezoid covering distance `dist`.

    Parameters
    ----------
    dist : float
        Total distance or displacement magnitude (>= 0).
    vmax : float
        Maximum allowed velocity (> 0).
    amax : float
        Maximum allowed acceleration (> 0).

    Returns
    -------
    t_acc : float
        Acceleration time duration (seconds).
    t_const : float
        Constant velocity phase duration (seconds).
    v_peak : float
        Peak velocity reached (<= vmax).
    total_time : float
        Total travel time duration (2 * t_acc + t_const).
    """
    if dist < 1e-12 or vmax <= 0.0 or amax <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    t_acc = vmax / amax
    d_acc = 0.5 * amax * (t_acc ** 2)

    if 2.0 * d_acc >= dist:  # Triangular profile
        t_acc = math.sqrt(dist / amax)
        v_peak = amax * t_acc
        t_const = 0.0
    else:  # Trapezoidal profile with cruise phase
        v_peak = vmax
        t_const = (dist - 2.0 * d_acc) / vmax

    total_time = 2.0 * t_acc + t_const
    return t_acc, t_const, v_peak, total_time


def scalar_trap_unit_progress(
    t: float,
    t_acc: float,
    t_const: float,
    v_peak: float,
    amax: float,
    t_target: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Evaluate (s, s_dot, s_ddot) for a unit-distance trapezoid at time t.

    If t_target > (2 * t_acc + t_const), cruise is extended to stretch to t_target.

    Parameters
    ----------
    t : float
        Current time in seconds.
    t_acc : float
        Acceleration phase duration.
    t_const : float
        Constant velocity phase duration.
    v_peak : float
        Peak velocity.
    amax : float
        Maximum acceleration.
    t_target : float, optional
        Target stretched duration.

    Returns
    -------
    s : float
        Normalized position progress in [0, 1].
    s_dot : float
        First time derivative of progress (1/s).
    s_ddot : float
        Second time derivative of progress (1/s^2).
    """
    if t <= 0.0:
        return 0.0, 0.0, 0.0

    t_tot = 2.0 * t_acc + t_const
    extra = (t_target - t_tot) if (t_target is not None and t_target > t_tot) else 0.0
    t_eff = t_tot + extra

    if t >= t_eff or t_eff <= 0.0:
        return 1.0, 0.0, 0.0

    t1 = t_acc
    t2 = t_acc + (t_const + extra)

    if t < t1:  # Acceleration phase
        a = amax
        v = a * t
        x = 0.5 * a * (t ** 2)
    elif t < t2:  # Cruise phase
        a = 0.0
        v = v_peak
        x = 0.5 * amax * (t_acc ** 2) + v_peak * (t - t_acc)
    else:  # Deceleration phase
        td = t - t2
        a = -amax
        v = v_peak - amax * td
        x = (
            0.5 * amax * (t_acc ** 2)
            + v_peak * (t_const + extra)
            + v_peak * td
            - 0.5 * amax * (td ** 2)
        )

    denom = v_peak * (t_const + extra) + amax * (t_acc ** 2)
    if denom <= 0.0:
        return 1.0, 0.0, 0.0

    return x / denom, v / denom, a / denom
