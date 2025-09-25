"""
SegmentedSE3Trap – analytic twist and acceleration, no finite differences.

Revision log 2025‑06‑25
• Pre‑compute the 6‑vector ξ = log6(A⁻¹B) for every segment.
• During runtime we evaluate σ(t), σ̇(t), σ̈(t) from the trapezoid coefficients.
• Pose:        T(t) = A · exp6(σ ξ)
• Spatial twist:    V(t) = Ad_{T(t)} (σ̇ ξ)
• Spatial accel:    Ẇ(t) = Ad_{T(t)} (σ̈ ξ)     (cross-term vanishes because ξ is constant)
  (If you prefer the body frame, drop the Ad_{T}.)
• This removes the recursion and the finite‑difference error entirely.

The rest of the public API (constructor, addWayPoint, computeTime, getMotionLaw)
remains identical.
"""

import math
import numpy as np
import pinocchio as pin
# ---------- helpers ---------------------------------------------------------

def _trapezoid_coeffs(dist, vmax, amax):
    """Return (t_acc, t_const, v_peak, total_time) for a 1‑D trapezoid."""
    if dist < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    t_acc = vmax / amax
    d_acc = 0.5 * amax * t_acc ** 2
    if 2 * d_acc >= dist:  # triangular
        t_acc = math.sqrt(dist / amax)
        v_peak = amax * t_acc
        t_const = 0.0
    else:  # full trapezoid
        v_peak = vmax
        t_const = (dist - 2 * d_acc) / vmax
    T = 2 * t_acc + t_const
    return t_acc, t_const, v_peak, T


def _scalar_trap(t, t_acc, t_const, v_peak, amax):
    """Return (s, s_dot, s_ddot) progress for a *unit distance* trapezoid."""
    if t <= 0.0:
        return 0.0, 0.0, 0.0
    T = 2 * t_acc + t_const
    if t >= T:
        return 1.0, 0.0, 0.0
    # accel
    if t < t_acc:
        a = amax
        v = a * t
        x = 0.5 * a * t ** 2
    # cruise
    elif t < t_acc + t_const:
        a = 0.0
        v = v_peak
        x = 0.5 * amax * t_acc ** 2 + v_peak * (t - t_acc)
    # decel
    else:
        td = t - (t_acc + t_const)
        a = -amax
        v = v_peak - amax * td
        x = 0.5 * amax * t_acc ** 2 + v_peak * t_const + v_peak * td - 0.5 * amax * td ** 2
    return x / (v_peak * t_const + amax * t_acc ** 2), v / (v_peak * t_const + amax * t_acc ** 2), a / (
        v_peak * t_const + amax * t_acc ** 2)


class SegmentedSE3Trap:
    """Multi‑way‑point trapezoidal SE3 planner with analytic twist & accel."""

    def __init__(self, vlin_max, vang_max, alin_max, aang_max):
        self.vlin_max = float(vlin_max)
        self.vang_max = float(vang_max)
        self.alin_max = float(alin_max)
        self.aang_max = float(aang_max)
        self._wps = []
        self._segments = []
        self._T_tot = 0.0

    # -- authoring ---------------------------------------------------------
    def addWayPoint(self, T: pin.SE3):
        self._wps.append(T.copy())

    # -- planning ----------------------------------------------------------
    def computeTime(self):
        if len(self._wps) < 2:
            raise RuntimeError("Need at least two way‑points")
        self._segments.clear()
        t_offset = 0.0
        for A, B in zip(self._wps[:-1], self._wps[1:]):
            # distances ----------------------------------------------------
            pA, pB = A.translation, B.translation
            d_lin = np.linalg.norm(pB - pA)
            Rrel = A.rotation.T @ B.rotation
            d_ang = np.linalg.norm(pin.log3(Rrel))

            # trapezoids ---------------------------------------------------
            tl_acc, tl_const, vl_peak, T_lin = _trapezoid_coeffs(d_lin, self.vlin_max, self.alin_max)
            ta_acc, ta_const, va_peak, T_ang = _trapezoid_coeffs(d_ang, self.vang_max, self.aang_max)
            T_seg = max(T_lin, T_ang)  # synchronise stop

            # constant screw between A and B
            xi = pin.log6(A.inverse() * B)  # 6‑vector
            self._segments.append({
                'A': A, 'B': B, 'xi': xi,
                't_start': t_offset,
                'T_seg': T_seg,
                # scalar profile params (pick slower axis)
                't_acc': tl_acc if T_lin >= T_ang else ta_acc,
                't_const': tl_const if T_lin >= T_ang else ta_const,
                'v_peak': vl_peak if T_lin >= T_ang else va_peak,
                'amax': self.alin_max if T_lin >= T_ang else self.aang_max,
            })
            t_offset += T_seg
        self._T_tot = t_offset
        return self._T_tot

    # -- query -------------------------------------------------------------
    def getMotionLaw(self, t):
        if len(self._segments) == 0:
            raise RuntimeError("Run computeTime() first")
        if t <= 0.0:
            zero = pin.Motion.Zero().vector
            return self._wps[0].copy(), zero, zero
        if t >= self._T_tot:
            zero = pin.Motion.Zero().vector
            return self._wps[-1].copy(), zero, zero

        for seg in self._segments:
            if seg['t_start'] <= t < seg['t_start'] + seg['T_seg']:
                tau = t - seg['t_start']
                s, s_dot, s_ddot = _scalar_trap(tau, seg['t_acc'], seg['t_const'], seg['v_peak'], seg['amax'])
                T_now = seg['A'] * pin.exp6(s * seg['xi'])

                # body twist / accel
                V_body = s_dot * seg['xi']
                A_body = s_ddot * seg['xi']

                # spatial (world) twist / accel
                V_spat = T_now.act(pin.Motion(V_body)).vector
                A_spat = T_now.act(pin.Motion(A_body)).vector
                return T_now, V_spat, A_spat
        raise RuntimeError("Time lookup failed")

    def publishPath(self):
        """
        Publish a poly-line (polygonal chain) that connects the stored way-points.

        Parameters
        ----------
        viz_viewer : meshcat.Visualizer
            The MeshCat viewer into which the path is drawn (e.g. ``viz.viewer``).
        name : str, default "path"
            Sub-node name under the viewer where the object is stored.
        color : int, default 0x0080ff
            RGB hex colour for the line.
        linewidth : float, default 4
            Line width in pixels.
        """
        if len(self._wps) < 2:
            raise RuntimeError("Need at least two way-points to publish a path")

        # Collect the translations of every way-point: shape (N, 3)
        return np.vstack([T.translation for T in self._wps])


