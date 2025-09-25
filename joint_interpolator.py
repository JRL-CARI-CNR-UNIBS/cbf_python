import math
import numpy as np

# ---------- helpers ---------------------------------------------------------

def _trapezoid_coeffs(dist, vmax, amax):
    """Return (t_acc, t_const, v_peak, total_time) for a 1-D trapezoid covering 'dist'."""
    if dist < 1e-12 or vmax <= 0.0 or amax <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    t_acc = vmax / amax
    d_acc = 0.5 * amax * t_acc ** 2
    if 2 * d_acc >= dist:  # triangular
        t_acc = math.sqrt(dist / amax)
        v_peak = amax * t_acc
        t_const = 0.0
    else:  # trapezoid
        v_peak = vmax
        t_const = (dist - 2 * d_acc) / vmax
    T = 2 * t_acc + t_const
    return t_acc, t_const, v_peak, T


def _scalar_trap_unit_progress(t, t_acc, t_const, v_peak, amax, T_target=None):
    """
    Evaluate (s, s_dot, s_ddot) for a *unit distance* trapezoid at time t.
    If T_target > (2*t_acc + t_const), we insert extra cruise to stretch to T_target.
    """
    if t <= 0.0:
        return 0.0, 0.0, 0.0

    T = 2 * t_acc + t_const
    if T_target is not None and T_target > T:
        extra = T_target - T
    else:
        extra = 0.0

    T_eff = T + extra
    if t >= T_eff or T_eff <= 0.0:
        return 1.0, 0.0, 0.0

    # piecewise with possibly-extended cruise
    t1 = t_acc
    t2 = t_acc + (t_const + extra)

    if t < t1:  # accel
        a = amax
        v = a * t
        x = 0.5 * a * t ** 2
    elif t < t2:  # cruise (possibly stretched)
        a = 0.0
        v = v_peak
        x = 0.5 * amax * t_acc ** 2 + v_peak * (t - t_acc)
    else:  # decel
        td = t - t2
        a = -amax
        v = v_peak - amax * td
        x = 0.5 * amax * t_acc ** 2 + v_peak * (t_const + extra) + v_peak * td - 0.5 * amax * td ** 2

    # total signed distance for unit profile is 1, so denom is total length of the *unit* trap:
    denom = v_peak * (t_const + extra) + amax * t_acc ** 2
    if denom <= 0.0:
        return 1.0, 0.0, 0.0
    s = x / denom
    s_dot = v / denom
    s_ddot = a / denom
    return s, s_dot, s_ddot



class SegmentedJointTrap:
    """
    Multi-way-point joint-space trapezoidal planner with shared timing.

    For each segment [qA -> qB]:
      1) Choose common times (t_acc, t_const) so the slowest joint runs as fast as allowed
         under per-joint (vmax, amax), and total time is minimized among schedules that
         share timing across joints.
      2) Each joint uses reduced accel/vel (<= its limits) so that all finish together.
      3) At query time, compute the normalized unit-distance progress (s, s_dot, s_ddot)
         ONCE using the shared times, then scale by each joint's dq.
    """

    def __init__(self, Dq_max: np.array, DDq_max: np.array):
        self.Dq_max = np.asarray(Dq_max, dtype=float).copy()   # per-joint vmax
        self.DDq_max = np.asarray(DDq_max, dtype=float).copy() # per-joint amax

        if self.Dq_max.shape != self.DDq_max.shape:
            raise ValueError("Dq_max and DDq_max must have the same shape")
        if np.any(self.Dq_max <= 0.0) or np.any(self.DDq_max <= 0.0):
            raise ValueError("All limits must be positive")

        self._dof = self.Dq_max.size
        self._wps = []
        self._segments = []
        self._T_tot = 0.0

        # lazy caches for fast segment lookup
        self._t_starts = None
        self._t_ends = None
        self._last_seg_idx = 0

    # -- authoring ---------------------------------------------------------
    def addWayPoint(self, q: np.array):
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self._dof:
            raise ValueError(f"Way-point has dim {q.size}, expected {self._dof}")
        self._wps.append(q.copy())

    # -- planning ----------------------------------------------------------
    def computeTime(self):
        """
        Build segments with shared timing per segment.

        For each segment with distances d_i = |Δq_i| and limits (vmax_i, amax_i):
          A = max_i d_i / amax_i
          B = max_i d_i / vmax_i
          y = max(sqrt(A), B)           # y = t_acc + t_const
          x = A / y                     # x = t_acc
          T_seg = x + y
        Per-joint used kinematics:
          a_used[i] = d_i / (x * y)     <= amax_i
          v_peak[i] = d_i / y           <= vmax_i
        """
        if len(self._wps) < 2:
            raise RuntimeError("Need at least two way-points")

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
                # degenerate zero-time segment
                self._segments.append({
                    'qA': qA, 'qB': qB, 'dq': dq,
                    't_start': t_offset,
                    'T_seg': 0.0,
                    't_acc_g': 0.0,
                    't_const_g': 0.0,
                    'a_used': np.zeros(self._dof),
                    'v_peak': np.zeros(self._dof),
                    'dist': abs_dq,
                })
                # t_offset unchanged
                continue

            # Compute common times
            A = float(np.max(abs_dq / self.DDq_max))  # max_i d_i / amax_i
            B = float(np.max(abs_dq / self.Dq_max))   # max_i d_i / vmax_i

            y = max(math.sqrt(A), B)  # y = t_acc + t_const
            # Guard against rare numerical issues (A could be 0 if all abs_dq==0, handled above)
            if y <= 0.0:
                x = 0.0
            else:
                x = A / y              # x = t_acc

            t_acc_g   = x
            t_const_g = y - x
            T_seg     = x + y

            # Per-joint actual kinematics (≤ limits by construction)
            with np.errstate(divide='ignore', invalid='ignore'):
                a_used = np.where(abs_dq >= eps, abs_dq / (x * y), 0.0)  # accel magnitude for each joint
                v_peak = np.where(abs_dq >= eps, abs_dq / y, 0.0)        # peak vel for each joint

            self._segments.append({
                'qA': qA, 'qB': qB, 'dq': dq,
                't_start': t_offset,
                'T_seg': T_seg,
                't_acc_g': t_acc_g,        # shared per segment
                't_const_g': t_const_g,    # shared per segment
                'a_used': a_used,          # per-joint accel actually used
                'v_peak': v_peak,          # per-joint peak vel actually used
                'dist': abs_dq,
            })
            t_offset += T_seg

        self._T_tot = t_offset
        return self._T_tot

    # -- query -------------------------------------------------------------
    def getMotionLaw(self, t: float):
        """
        Return (q(t), qdot(t), qddot(t)) at absolute time t, using shared timing per segment.
        Computes normalized (s, s_dot, s_ddot) ONCE per query and scales by dq.
        """
        if len(self._segments) == 0:
            raise RuntimeError("Run computeTime() first")

        if t <= 0.0:
            q0 = self._wps[0].copy()
            zero = np.zeros_like(q0)
            return q0, zero, zero

        if t >= self._T_tot:
            qN = self._wps[-1].copy()
            zero = np.zeros_like(qN)
            return qN, zero, zero

        # Lazy-build segment time arrays for O(log N) lookup
        if self._t_starts is None:
            nseg = len(self._segments)
            self._t_starts = np.empty(nseg, dtype=float)
            self._t_ends   = np.empty(nseg, dtype=float)
            for k, seg in enumerate(self._segments):
                t0 = seg['t_start']
                T  = seg['T_seg']
                self._t_starts[k] = t0
                self._t_ends[k]   = t0 + T
            self._last_seg_idx = 0

        # Hot-path cache (monotonic/near-monotonic queries)
        idx = self._last_seg_idx
        if not (self._t_starts[idx] <= t < self._t_ends[idx]):
            # Binary search
            idx = np.searchsorted(self._t_starts, t, side='right') - 1
            if idx < 0:
                idx = 0
            while idx + 1 < self._t_ends.size and t >= self._t_ends[idx]:
                idx += 1
            self._last_seg_idx = idx

        seg = self._segments[idx]
        t0 = seg['t_start']
        T_seg = seg['T_seg']

        # Degenerate zero-time segment
        if T_seg == 0.0:
            q = seg['qB'].copy()
            z = np.zeros_like(q)
            return q, z, z

        tau = t - t0

        # Shared times
        x  = seg['t_acc_g']       # t_acc (common)
        tc = seg['t_const_g']     # t_const (common)
        y  = x + tc               # convenience: y = t_acc + t_const

        # Compute normalized unit-distance progress ONCE.
        # For a unit-distance profile with shared times, set:
        #   a0 = 1 / (x*y), vp0 = 1 / y
        s, s_dot, s_ddot = _scalar_trap_unit_progress(
            tau,
            x,
            tc,
            v_peak=1.0 / y if y > 0.0 else 0.0,
            amax=1.0 / (x * y) if (x > 0.0 and y > 0.0) else 0.0,
            T_target=None
        )

        # Scale to each joint
        dq   = seg['dq']
        qA   = seg['qA']
        q     = qA + s      * dq
        qdot  = s_dot  * dq
        qddot = s_ddot * dq

        return q, qdot, qddot

    def publishPath(self):
        """
        Return the stacked way-points as an (N, dof) array (polyline in joint space).
        """
        if len(self._wps) < 2:
            raise RuntimeError("Need at least two way-points to publish a path")
        return np.vstack(self._wps)
