"""
Cartesian SE3 multi-waypoint trapezoidal trajectory interpolator with analytic twist and acceleration.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
import pinocchio as pin

from cbf_python.trajectory.trapezoid import trapezoid_coeffs, scalar_trap_unit_progress


class SegmentedSE3Trap:
    """Multi-waypoint trapezoidal SE3 planner with analytic spatial twist and acceleration.

    Precomputes the 6-vector screw coordinates xi = log6(A^-1 * B) for each segment.
    Evaluates unit progress s(t), s_dot(t), s_ddot(t) analytically without finite differencing.
    """

    def __init__(
        self,
        vlin_max: float,
        vang_max: float,
        alin_max: float,
        aang_max: float,
    ) -> None:
        self.vlin_max = float(vlin_max)
        self.vang_max = float(vang_max)
        self.alin_max = float(alin_max)
        self.aang_max = float(aang_max)

        self._wps: List[pin.SE3] = []
        self._segments: List[Dict[str, Any]] = []
        self._T_tot: float = 0.0

    @property
    def total_time(self) -> float:
        """Total duration of the planned SE3 trajectory."""
        return self._T_tot

    def addWayPoint(self, T: pin.SE3) -> None:
        """Append an SE3 pose waypoint."""
        self._wps.append(T.copy())

    def computeTime(self) -> float:
        """Compute synchronized segment timings between consecutive SE3 poses.

        Returns
        -------
        float
            Total trajectory duration in seconds.
        """
        if len(self._wps) < 2:
            raise RuntimeError("Need at least two waypoints to plan an SE3 trajectory")

        self._segments.clear()
        t_offset = 0.0

        for A, B in zip(self._wps[:-1], self._wps[1:]):
            pA, pB = A.translation, B.translation
            d_lin = float(np.linalg.norm(pB - pA))
            Rrel = A.rotation.T @ B.rotation
            d_ang = float(np.linalg.norm(pin.log3(Rrel)))

            tl_acc, tl_const, vl_peak, T_lin = trapezoid_coeffs(d_lin, self.vlin_max, self.alin_max)
            ta_acc, ta_const, va_peak, T_ang = trapezoid_coeffs(d_ang, self.vang_max, self.aang_max)
            T_seg = max(T_lin, T_ang)

            # Constant screw coordinates between A and B
            xi = pin.log6(A.inverse() * B)

            self._segments.append({
                "A": A,
                "B": B,
                "xi": xi,
                "t_start": t_offset,
                "T_seg": T_seg,
                "t_acc": tl_acc if T_lin >= T_ang else ta_acc,
                "t_const": tl_const if T_lin >= T_ang else ta_const,
                "v_peak": vl_peak if T_lin >= T_ang else va_peak,
                "amax": self.alin_max if T_lin >= T_ang else self.aang_max,
            })
            t_offset += T_seg

        self._T_tot = t_offset
        return self._T_tot

    def getMotionLaw(self, t: float) -> Tuple[pin.SE3, np.ndarray, np.ndarray]:
        """Evaluate SE3 pose, spatial twist, and spatial acceleration at time t.

        Parameters
        ----------
        t : float
            Evaluation time in seconds.

        Returns
        -------
        Tuple[pin.SE3, np.ndarray, np.ndarray]
            (T(t), V_spatial(t), A_spatial(t))
        """
        if len(self._segments) == 0:
            raise RuntimeError("computeTime() must be called before querying motion law")

        if t <= 0.0:
            zero = pin.Motion.Zero().vector
            return self._wps[0].copy(), zero, zero

        if t >= self._T_tot:
            zero = pin.Motion.Zero().vector
            return self._wps[-1].copy(), zero, zero

        for seg in self._segments:
            t0 = seg["t_start"]
            T_seg = seg["T_seg"]
            if t0 <= t < t0 + T_seg:
                tau = t - t0
                s, s_dot, s_ddot = scalar_trap_unit_progress(
                    tau,
                    seg["t_acc"],
                    seg["t_const"],
                    seg["v_peak"],
                    seg["amax"],
                )
                T_now = seg["A"] * pin.exp6(s * seg["xi"])

                # Body twist and acceleration
                V_body = s_dot * seg["xi"]
                A_body = s_ddot * seg["xi"]

                # Convert to spatial (world) frame
                V_spat = T_now.act(pin.Motion(V_body)).vector
                A_spat = T_now.act(pin.Motion(A_body)).vector
                return T_now, V_spat, A_spat

        # Edge-case fallback for floating point equality at upper boundary
        zero = pin.Motion.Zero().vector
        return self._wps[-1].copy(), zero, zero

    def publishPath(self) -> np.ndarray:
        """Return waypoint translations stacked as an (N, 3) polyline array."""
        if len(self._wps) < 2:
            raise RuntimeError("Need at least two waypoints to publish path")
        return np.vstack([T.translation for T in self._wps])
