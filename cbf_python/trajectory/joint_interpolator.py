"""
Joint-space multi-waypoint trapezoidal trajectory interpolator with shared timing.
"""

from __future__ import annotations
import math
from typing import Sequence, List, Dict, Any, Tuple
import numpy as np

from cbf_python.trajectory.trapezoid import scalar_trap_unit_progress


class SegmentedJointTrap:
    """Multi-waypoint joint-space trapezoidal planner with synchronized joint timing.

    For each segment [qA -> qB]:
      1) Chooses common timing (t_acc, t_const) such that the slowest joint moves at its
         kinematic limit (vmax, amax) and total segment time is minimized.
      2) Scales down acceleration and peak velocity for faster joints so all joints arrive simultaneously.
      3) Evaluates unit progress (s, s_dot, s_ddot) once per query and scales by joint displacements.
    """

    def __init__(self, Dq_max: Sequence[float] | np.ndarray, DDq_max: Sequence[float] | np.ndarray) -> None:
        self.Dq_max = np.asarray(Dq_max, dtype=float).copy()
        self.DDq_max = np.asarray(DDq_max, dtype=float).copy()

        if self.Dq_max.shape != self.DDq_max.shape:
            raise ValueError("Dq_max and DDq_max must have the same shape")
        if np.any(self.Dq_max <= 0.0) or np.any(self.DDq_max <= 0.0):
            raise ValueError("All kinematic limits must be positive")

        self._dof: int = self.Dq_max.size
        self._wps: List[np.ndarray] = []
        self._segments: List[Dict[str, Any]] = []
        self._T_tot: float = 0.0

        # Lazy caches for O(log N) fast segment search
        self._t_starts: np.ndarray | None = None
        self._t_ends: np.ndarray | None = None
        self._last_seg_idx: int = 0

    @property
    def dof(self) -> int:
        """Number of degrees of freedom."""
        return self._dof

    @property
    def total_time(self) -> float:
        """Total duration of the planned trajectory."""
        return self._T_tot

    def addWayPoint(self, q: Sequence[float] | np.ndarray) -> None:
        """Add a waypoint to the trajectory."""
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        if q_arr.size != self._dof:
            raise ValueError(f"Waypoint dimension {q_arr.size} does not match expected {self._dof}")
        self._wps.append(q_arr.copy())

    def computeTime(self) -> float:
        """Compute synchronized segment timings across all joints.

        Returns
        -------
        float
            Total planned trajectory duration.
        """
        if len(self._wps) < 2:
            raise RuntimeError("Need at least two waypoints to compute a trajectory")

        self._segments.clear()
        self._t_starts = None
        self._t_ends = None
        self._last_seg_idx = 0

        t_offset = 0.0
        eps = 1e-12

        for qA, qB in zip(self._wps[:-1], self._wps[1:]):
            dq = (qB - qA).astype(float)
            abs_dq = np.abs(dq)

            if np.all(abs_dq < eps):
                # Degenerate zero-time segment
                self._segments.append({
                    "qA": qA, "qB": qB, "dq": dq,
                    "t_start": t_offset,
                    "T_seg": 0.0,
                    "t_acc_g": 0.0,
                    "t_const_g": 0.0,
                    "a_used": np.zeros(self._dof),
                    "v_peak": np.zeros(self._dof),
                    "dist": abs_dq,
                })
                continue

            # Compute synchronized timing for segment
            A = float(np.max(abs_dq / self.DDq_max))
            B = float(np.max(abs_dq / self.Dq_max))

            y = max(math.sqrt(A), B)  # y = t_acc + t_const
            x = (A / y) if y > 0.0 else 0.0  # x = t_acc

            t_acc_g = x
            t_const_g = y - x
            T_seg = x + y

            with np.errstate(divide="ignore", invalid="ignore"):
                a_used = np.where(abs_dq >= eps, abs_dq / (x * y), 0.0)
                v_peak = np.where(abs_dq >= eps, abs_dq / y, 0.0)

            self._segments.append({
                "qA": qA, "qB": qB, "dq": dq,
                "t_start": t_offset,
                "T_seg": T_seg,
                "t_acc_g": t_acc_g,
                "t_const_g": t_const_g,
                "a_used": a_used,
                "v_peak": v_peak,
                "dist": abs_dq,
            })
            t_offset += T_seg

        self._T_tot = t_offset
        return self._T_tot

    def getMotionLaw(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Query position, velocity, and acceleration at time t.

        Parameters
        ----------
        t : float
            Evaluation time in seconds.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (q, qdot, qddot) evaluated vectors of shape (dof,).
        """
        if len(self._segments) == 0:
            raise RuntimeError("computeTime() must be called before querying motion law")

        if t <= 0.0:
            q0 = self._wps[0].copy()
            zero = np.zeros_like(q0)
            return q0, zero, zero

        if t >= self._T_tot:
            qN = self._wps[-1].copy()
            zero = np.zeros_like(qN)
            return qN, zero, zero

        # Lazy cache segment start/end bounds for fast lookup
        if self._t_starts is None:
            nseg = len(self._segments)
            self._t_starts = np.empty(nseg, dtype=float)
            self._t_ends = np.empty(nseg, dtype=float)
            for k, seg in enumerate(self._segments):
                t0 = seg["t_start"]
                T = seg["T_seg"]
                self._t_starts[k] = t0
                self._t_ends[k] = t0 + T
            self._last_seg_idx = 0

        idx = self._last_seg_idx
        if not (self._t_starts[idx] <= t < self._t_ends[idx]):
            idx = int(np.searchsorted(self._t_starts, t, side="right") - 1)
            idx = max(0, idx)
            while idx + 1 < self._t_ends.size and t >= self._t_ends[idx]:
                idx += 1
            self._last_seg_idx = idx

        seg = self._segments[idx]
        t0 = seg["t_start"]
        T_seg = seg["T_seg"]

        if T_seg == 0.0:
            q = seg["qB"].copy()
            z = np.zeros_like(q)
            return q, z, z

        tau = t - t0
        x = seg["t_acc_g"]
        tc = seg["t_const_g"]
        y = x + tc

        s, s_dot, s_ddot = scalar_trap_unit_progress(
            tau,
            x,
            tc,
            v_peak=1.0 / y if y > 0.0 else 0.0,
            amax=1.0 / (x * y) if (x > 0.0 and y > 0.0) else 0.0,
            t_target=None,
        )

        dq = seg["dq"]
        qA = seg["qA"]
        q = qA + s * dq
        qdot = s_dot * dq
        qddot = s_ddot * dq

        return q, qdot, qddot

    def publishPath(self) -> np.ndarray:
        """Return all waypoints stacked as an (N, dof) array."""
        if len(self._wps) < 2:
            raise RuntimeError("Need at least two waypoints to publish path")
        return np.vstack(self._wps)
