"""
Speed and Separation Monitoring (SSM) Control Barrier Function (CBF) Kernels.

This module provides Numba-accelerated functions to compute the predictive
stopping-horizon separation distance, its analytic sensitivities (Jacobian),
Lie derivatives along control-affine system dynamics, and robust barrier inequality
constraints on robot joint accelerations.

Theoretical Principles:
    1. Stopping-Horizon Trajectory Model:
       Given the relative distance, projected velocities, and estimated human
       acceleration along the line of sight, the system predicts the minimum separation
       distance d_min that would occur during an open-loop reaction and braking procedure.
       - Reaction phase: duration Tr, robot moves at constant velocity v_r.
       - Braking phase: uniform deceleration at capability a_s until coming to a complete stop.
    2. Candidate Extrema:
       By the extreme value theorem, d_min occurs either at the evaluation boundaries
       or at stationary points where the relative closing speed cancels out:
       - t0: initial causality lower bound (t0 >= tc)
       - t1: stationary point during reaction phase [t0, Tr]
       - t2: reaction time Tr (braking onset)
       - t3: stationary point during braking phase (Tr, Tf]
       - t4: terminal full stop instant Tf = Tr + |v_r| / a_s
    3. Robust Barrier Formulation:
       The barrier function is defined as h = d_min - C_safe.
       The barrier condition L_f h + L_g h * mu >= -gamma * (h - margin) restricts
       the commanded Cartesian acceleration to guarantee forward invariance of the safe set.
"""

import math
import numpy as np
from numba import njit, float64, boolean



# ------------------------------------------------------------
# 1) Dynamics: Drift Vector f(x) and Input Matrix g(x)
# ------------------------------------------------------------
@njit(cache=True)
def range_state_derivative_numba(v_r: np.ndarray, v_h: np.ndarray):
    """
    Compute state derivative components for relative kinematic dynamics.

    State vector x in R^12: [p_r (3), p_h (3), v_r (3), v_h (3)]
    Control input mu in R^3: robot Cartesian acceleration (dv_r/dt)

    Parameters:
        v_r: Robot Cartesian velocity vector in R^3 [m/s]
        v_h: Human Cartesian velocity vector in R^3 [m/s]

    Returns:
        f: Drift vector in R^12 representing unforced state velocity
        g: Input mapping matrix in R^(12x3) mapping Cartesian acceleration to state derivative
    """
    f = np.zeros(12, dtype=np.float64)
    f[0:3] = v_r
    f[3:6] = v_h

    g = np.zeros((12, 3), dtype=np.float64)
    g[6, 0] = 1.0
    g[7, 1] = 1.0
    g[8, 2] = 1.0
    return f, g


# ------------------------------------------------------------
# 2) Core Distance and Stopping-Horizon Extrema
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def dmin_and_jacobian_numba(
    d: float,
    v_r: float,
    v_h: float,
    a_h: float,
    tr: float,
    a_max: float,
    atol: float,
    tc: float = 0.0,
):
    """
    Calculate the predicted minimum separation distance and its analytic Jacobian.

    Evaluates prospective distance along the line-of-sight stopping trajectory:
        Phase 1: tau in [t0, tr] (Reaction Phase)
            d(tau) = d + (v_r - v_h)*tau - 0.5*a_h*tau^2
        Phase 2: tau in (tr, tf] (Braking Phase)
            d(tau) = d + (v_r - v_h)*tau - 0.5*a_h*tau^2 + 0.5*a_max*(tau - tr)^2

    Parameters:
        d: Current distance between robot and human keypoint [m]
        v_r: Robot projected velocity along line of sight [m/s] (< 0 approaches human)
        v_h: Human projected velocity along line of sight [m/s] (> 0 approaches robot)
        a_h: Human projected acceleration along line of sight [m/s^2]
        tr: System reaction time [s] (sensing, communication, and actuation delay)
        a_max: Robot maximum braking deceleration capability [m/s^2] (positive scalar)
        atol: Numerical tolerance for zero comparisons and deduplication
        tc: Control cycle sampling period [s]; enforces causality lower bound (t0 >= tc)

    Returns:
        d_min: Minimum separation distance over the prospective horizon [m]
        jac: Sensitivity vector [dd_min/dd, dd_min/dv_r, dd_min/dv_h, dd_min/da_h]
    """
    if a_max <= 0.0:
        raise ValueError("a_max must be positive.")

    # Lower bound on prospective evaluation horizon:
    # Under discrete zero-order hold control, the earliest instant the current acceleration
    # command possesses control authority is tau = tc. When tc=0 (e.g. continuous tests), t0=0.
    t0 = max(tc, 0.0)

    # Reaction horizon boundary
    t2 = tr

    # Braking duration until complete standstill (active only if robot approaches human, v_r < 0)
    t_dec = -v_r / a_max
    if t_dec < 0.0:
        t_dec = 0.0
    t4 = t2 + t_dec  # Terminal stopping time Tf

    m = a_h - a_max
    v_diff = v_r - v_h

    # Stationary point during reaction phase: dot{d}(tau) = 0 <=> v_diff - a_h * tau = 0
    has_t1 = False
    t1 = 0.0
    if abs(a_h) > atol:
        t1_raw = v_diff / a_h
        if (t0 + atol) < t1_raw < (t2 - atol):
            t1 = t1_raw
            has_t1 = True

    # Stationary point during braking phase: dot{d}(tau) = 0 <=> v_diff - a_h*tau + a_max*(tau - tr) = 0
    has_t3 = False
    t3 = 0.0
    if abs(m) > atol and (t4 - t2) > atol:
        t3_raw = (v_diff - a_max * t2) / m
        if (t2 + atol) < t3_raw < (t4 - atol):
            t3 = t3_raw
            has_t3 = True

    # Collect candidate extrema
    uniq = np.empty(4, dtype=np.float64)
    n = 0

    def add_candidate(tt):
        nonlocal n
        for k in range(n):
            if abs(tt - uniq[k]) <= atol:
                return
        uniq[n] = tt
        n += 1

    if v_r < 0.0:
        # Robot moving toward human: braking phase is relevant
        add_candidate(t0)
        add_candidate(t4)
        if has_t1:
            add_candidate(t1)
        if has_t3:
            add_candidate(t3)
    else:
        # Robot moving away or stationary: evaluation bounded by reaction phase
        add_candidate(t0)
        add_candidate(t2)
        if has_t1:
            add_candidate(t1)

    # Evaluate distance trajectory at each candidate instant
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        tt = uniq[i]
        if tt <= t2 + atol:
            vals[i] = d + v_diff * tt - 0.5 * a_h * (tt * tt)
        else:
            vals[i] = d + v_diff * tt - 0.5 * a_h * (tt * tt) + 0.5 * a_max * (tt - t2) * (tt - t2)

    # Identify active global minimizer
    i_min = 0
    best = vals[0]
    for i in range(1, n):
        if vals[i] < best:
            best = vals[i]
            i_min = i

    t_star = uniq[i_min]
    d_min = best

    # Analytic sensitivity of d_min w.r.t [d, v_r, v_h, a_h]
    # By Danskin's Envelope Theorem, at stationary extrema (dot{d} = 0), total derivative equals partial derivative.
    jac = np.zeros(4, dtype=np.float64)

    if abs(t_star - t0) <= atol:
        jac[0] = 1.0
        if t0 > atol:
            jac[1] = t0
            jac[2] = -t0
            jac[3] = -0.5 * (t0 * t0)
        return d_min, jac

    if has_t1 and abs(t_star - t1) <= atol:
        jac[0] = 1.0
        jac[1] = t1
        jac[2] = -t1
        jac[3] = -0.5 * (t1 * t1)
        return d_min, jac

    if abs(t_star - t2) <= atol:
        jac[0] = 1.0
        jac[1] = t2
        jac[2] = -t2
        jac[3] = -0.5 * (t2 * t2)
        return d_min, jac

    if has_t3 and abs(t_star - t3) <= atol:
        jac[0] = 1.0
        jac[1] = t3
        jac[2] = -t3
        jac[3] = -0.5 * (t3 * t3)
        return d_min, jac

    # Terminal stopping boundary instant t4 = tr - v_r / a_max
    if abs(t_star - t4) <= atol:
        t4_sq = t4 * t4
        jac[0] = 1.0
        jac[1] = t4
        jac[2] = -t4
        jac[3] = -0.5 * t4_sq
        if t_dec > atol:
            dotd_t4_now = v_diff - a_max * t2 - m * t4
            jac[1] += -dotd_t4_now / a_max
        return d_min, jac

    return d_min, jac


# ------------------------------------------------------------
# 3) Control Barrier Function Evaluation
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def h_and_jacobian_numba(
    d: float,
    v_r: float,
    v_h: float,
    a_h: float,
    tr: float,
    a_max: float,
    C: float,
    atol: float,
    tc: float = 0.0,
):
    """
    Compute the Control Barrier Function h and its gradient w.r.t [d, v_r, v_h, a_h].

    The safe set C_safe is defined as:
        C_safe = { x | h(x) >= 0 }
    where:
        h = d_min - C
    with C representing the minimum required clearance distance [m].

    Parameters:
        d: Current separation distance [m]
        v_r, v_h: Projected velocities along line of sight [m/s]
        a_h: Projected human acceleration [m/s^2]
        tr: Reaction time [s]
        a_max: Braking deceleration capability [m/s^2]
        C: Safety separation threshold [m]
        atol: Numerical comparison tolerance
        tc: Sampling period [s]

    Returns:
        h_val: Barrier margin value [m] (h >= 0 indicates membership in safe set)
        h_jac: Barrier gradient vector [dh/dd, dh/dv_r, dh/dv_h, dh/da_h]
    """
    d_min, dist_jac = dmin_and_jacobian_numba(d, v_r, v_h, a_h, tr, a_max, atol, tc)

    h_val = d_min - C
    h_jac = np.empty(4, dtype=np.float64)
    h_jac[0] = dist_jac[0]
    h_jac[1] = dist_jac[1]
    h_jac[2] = dist_jac[2]
    h_jac[3] = dist_jac[3]

    return h_val, h_jac


# ------------------------------------------------------------
# 4) Geometric Jacobian blocks
# ------------------------------------------------------------
@njit(cache=True)
def jacobian_psi_numba(p_r: np.ndarray, p_h: np.ndarray, v_lin: np.ndarray, v_human: np.ndarray):
    """
    Numba version of jacobian_psi.
    p_r, p_h: (3,)
    v_lin, v_human: (3,)
    Returns:
        J (4,12)
    Layout:
      row0: [ u^T | -u^T | 0 | 0 ]
      row1: [ (v_lin P) | -(v_lin P) | u^T | 0 ]
      row2: [ (v_hum P) | -(v_hum P) | 0 | u^T ]
      row3: zeros
    """
    diff = p_r - p_h
    norm = np.sqrt(np.dot(diff, diff))
    if norm == 0.0:
        u = np.zeros(3, dtype=np.float64)
        u[0] = 1.0
    else:
        u = diff / norm

    P = np.eye(3, dtype=np.float64)
    P[0, 0] -= u[0] * u[0]
    P[0, 1] -= u[0] * u[1]
    P[0, 2] -= u[0] * u[2]
    P[1, 0] -= u[1] * u[0]
    P[1, 1] -= u[1] * u[1]
    P[1, 2] -= u[1] * u[2]
    P[2, 0] -= u[2] * u[0]
    P[2, 1] -= u[2] * u[1]
    P[2, 2] -= u[2] * u[2]

    vlinP = np.zeros(3, dtype=np.float64)
    vlinP[0] = v_lin[0] * P[0, 0] + v_lin[1] * P[1, 0] + v_lin[2] * P[2, 0]
    vlinP[1] = v_lin[0] * P[0, 1] + v_lin[1] * P[1, 1] + v_lin[2] * P[2, 1]
    vlinP[2] = v_lin[0] * P[0, 2] + v_lin[1] * P[1, 2] + v_lin[2] * P[2, 2]

    vhumP = np.zeros(3, dtype=np.float64)
    vhumP[0] = v_human[0] * P[0, 0] + v_human[1] * P[1, 0] + v_human[2] * P[2, 0]
    vhumP[1] = v_human[0] * P[0, 1] + v_human[1] * P[1, 1] + v_human[2] * P[2, 1]
    vhumP[2] = v_human[0] * P[0, 2] + v_human[1] * P[1, 2] + v_human[2] * P[2, 2]

    J = np.zeros((4, 12), dtype=np.float64)
    J[0, 0:3] = u
    J[0, 3:6] = -u
    J[1, 0:3] = vlinP
    J[1, 3:6] = -vlinP
    J[1, 6:9] = u
    J[2, 0:3] = vhumP
    J[2, 3:6] = -vhumP
    J[2, 9:12] = u
    return J


# ------------------------------------------------------------
# 5) Fast contractions: (Jpsi @ f) and (Jpsi @ g)
# ------------------------------------------------------------
@njit(cache=True)
def jacobian_psi_times_fg_fast_numba(
    p_r: np.ndarray, p_h: np.ndarray,
    v_r: np.ndarray, v_h: np.ndarray,
    a_h_vec: np.ndarray,
    atol: float = 1e-12,
):
    """
    Returns:
        Jpsi_f: (4,)
        Jpsi_g: (4,3)
    """
    r = p_r - p_h
    d = max(np.sqrt(np.dot(r, r)), 0.001)

    if d <= atol:
        u = np.copy(v_r)
        nrm = np.sqrt(np.dot(u, u))
        if nrm <= atol:
            u[0] = 1.0
            u[1] = 0.0
            u[2] = 0.0
        else:
            u /= nrm
        if d < atol:
            d = atol
    else:
        u = r / d

    v_diff = v_r - v_h

    vr_rel = u[0] * v_r[0] + u[1] * v_r[1] + u[2] * v_r[2]
    vh_rel = u[0] * v_h[0] + u[1] * v_h[1] + u[2] * v_h[2]
    ah_rel = u[0] * a_h_vec[0] + u[1] * a_h_vec[1] + u[2] * a_h_vec[2]

    # Positional gradient vectors (tangential vector / distance)
    grad_pr_vr = (v_r - u * vr_rel) / d
    grad_pr_vh = (v_h - u * vh_rel) / d
    grad_pr_ah = (a_h_vec - u * ah_rel) / d

    Jpsi_f = np.zeros(4, dtype=np.float64)
    Jpsi_f[0] = u[0] * v_diff[0] + u[1] * v_diff[1] + u[2] * v_diff[2]
    Jpsi_f[1] = grad_pr_vr[0] * v_diff[0] + grad_pr_vr[1] * v_diff[1] + grad_pr_vr[2] * v_diff[2]
    Jpsi_f[2] = grad_pr_vh[0] * v_diff[0] + grad_pr_vh[1] * v_diff[1] + grad_pr_vh[2] * v_diff[2]
    Jpsi_f[3] = grad_pr_ah[0] * v_diff[0] + grad_pr_ah[1] * v_diff[1] + grad_pr_ah[2] * v_diff[2]

    Jpsi_g = np.zeros((4, 3), dtype=np.float64)
    Jpsi_g[1, 0] = u[0]
    Jpsi_g[1, 1] = u[1]
    Jpsi_g[1, 2] = u[2]

    return Jpsi_f, Jpsi_g


# ------------------------------------------------------------
# 6) Lie derivatives
# ------------------------------------------------------------
@njit(cache=True)
def compute_g_Lie_terms_numba(
    translation_bt: np.ndarray,
    obs_pos: np.ndarray,
    vel_lineare: np.ndarray,
    v_obs: np.ndarray,
    a_h: float,
    Tr: float,
    a_s: float,
    C: float,
    atol: float = 1e-12,
):
    """
    Compute:
      - g (12x3)
      - Lie_f_h (scalar)
      - Lie_g_h (3,)
    """
    r0 = translation_bt[0] - obs_pos[0]
    r1 = translation_bt[1] - obs_pos[1]
    r2 = translation_bt[2] - obs_pos[2]
    d2 = r0*r0 + r1*r1 + r2*r2

    if d2 <= atol*atol:
        nrm2 = vel_lineare[0]*vel_lineare[0] + vel_lineare[1]*vel_lineare[1] + vel_lineare[2]*vel_lineare[2]
        if nrm2 <= atol*atol:
            u0, u1, u2 = 1.0, 0.0, 0.0
        else:
            inv = 1.0 / np.sqrt(nrm2)
            u0 = vel_lineare[0] * inv
            u1 = vel_lineare[1] * inv
            u2 = vel_lineare[2] * inv
        d = np.sqrt(d2) if d2 > 0.0 else atol
    else:
        inv = 1.0 / np.sqrt(d2)
        u0, u1, u2 = r0*inv, r1*inv, r2*inv
        d = 1.0 / inv

    v_rel = u0*vel_lineare[0] + u1*vel_lineare[1] + u2*vel_lineare[2]
    v_h   = u0*v_obs[0]       + u1*v_obs[1]       + u2*v_obs[2]

    h_val, Jh_psi = h_and_jacobian_numba(d, v_rel, v_h, a_h, Tr, a_s, C, atol)

    zero3 = np.zeros(3, dtype=np.float64)
    Jpsi_f, Jpsi_g = jacobian_psi_times_fg_fast_numba(
        p_r=translation_bt, p_h=obs_pos, v_r=vel_lineare, v_h=v_obs, a_h_vec=zero3, atol=atol
    )

    Lie_f_h = Jh_psi[0]*Jpsi_f[0] + Jh_psi[1]*Jpsi_f[1] + Jh_psi[2]*Jpsi_f[2] + Jh_psi[3]*Jpsi_f[3]

    Lie_g_h = np.zeros(3, dtype=np.float64)
    Lie_g_h[0] = Jh_psi[0]*Jpsi_g[0,0] + Jh_psi[1]*Jpsi_g[1,0] + Jh_psi[2]*Jpsi_g[2,0] + Jh_psi[3]*Jpsi_g[3,0]
    Lie_g_h[1] = Jh_psi[0]*Jpsi_g[0,1] + Jh_psi[1]*Jpsi_g[1,1] + Jh_psi[2]*Jpsi_g[2,1] + Jh_psi[3]*Jpsi_g[3,1]
    Lie_g_h[2] = Jh_psi[0]*Jpsi_g[0,2] + Jh_psi[1]*Jpsi_g[1,2] + Jh_psi[2]*Jpsi_g[2,2] + Jh_psi[3]*Jpsi_g[3,2]

    g = np.zeros((12, 3), dtype=np.float64)
    g[6, 0] = 1.0
    g[7, 1] = 1.0
    g[8, 2] = 1.0

    return g, Lie_f_h, Lie_g_h


@njit(cache=True, fastmath=True)
def compute_h_lie_and_grad_numba(
    translation_bt: np.ndarray,
    obs_pos: np.ndarray,
    vel_lineare: np.ndarray,
    v_obs: np.ndarray,
    Tr: float,
    a_s: float,
    C: float,
    obs_acc: np.ndarray,
    atol: float,
    tc: float = 0.0,
):
    """
    Compute barrier margin h, drift Lie derivative L_f h, control Lie derivative L_g h,
    and full feature gradient Jh_psi = grad_psi h = [dh/dd, dh/dvr, dh/dvh, dh/dah].
    """
    r0 = translation_bt[0] - obs_pos[0]
    r1 = translation_bt[1] - obs_pos[1]
    r2 = translation_bt[2] - obs_pos[2]
    d = np.sqrt(r0 * r0 + r1 * r1 + r2 * r2)

    # Unit line-of-sight direction vector u pointing from human to robot
    if d <= atol:
        u0, u1, u2 = 1.0, 0.0, 0.0
        if d < atol:
            d = atol
    else:
        invd = 1.0 / d
        u0, u1, u2 = r0 * invd, r1 * invd, r2 * invd

    # Project velocities and accelerations onto line-of-sight vector
    v_r = u0 * vel_lineare[0] + u1 * vel_lineare[1] + u2 * vel_lineare[2]
    v_h = u0 * v_obs[0] + u1 * v_obs[1] + u2 * v_obs[2]
    a_h = u0 * obs_acc[0] + u1 * obs_acc[1] + u2 * obs_acc[2]

    # Evaluate predictive stopping-horizon barrier and its feature-space gradient
    h, Jh_psi = h_and_jacobian_numba(d, v_r, v_h, a_h, Tr, a_s, C, atol, tc)

    # Fast contraction of feature Jacobian with system drift f and input matrix g
    Jpsi_f, Jpsi_g = jacobian_psi_times_fg_fast_numba(translation_bt, obs_pos, vel_lineare, v_obs, obs_acc, atol)

    # Drift Lie derivative: L_f h = sum_k (dh/dpsi_k) * (dpsi_k/dt)_drift
    Lie_f_h = Jh_psi[0] * Jpsi_f[0] + Jh_psi[1] * Jpsi_f[1] + Jh_psi[2] * (Jpsi_f[2] + a_h) + Jh_psi[3] * Jpsi_f[3]

    # Control Lie derivative: L_g h = sum_k (dh/dpsi_k) * (Jpsi_g)_k
    Lie_g_h = np.zeros(3, dtype=np.float64)
    Lie_g_h[0] += Jh_psi[0] * Jpsi_g[0, 0] + Jh_psi[1] * Jpsi_g[1, 0] + Jh_psi[2] * Jpsi_g[2, 0] + Jh_psi[3] * Jpsi_g[3, 0]
    Lie_g_h[1] += Jh_psi[0] * Jpsi_g[0, 1] + Jh_psi[1] * Jpsi_g[1, 1] + Jh_psi[2] * Jpsi_g[2, 1] + Jh_psi[3] * Jpsi_g[3, 1]
    Lie_g_h[2] += Jh_psi[0] * Jpsi_g[0, 2] + Jh_psi[1] * Jpsi_g[1, 2] + Jh_psi[2] * Jpsi_g[2, 2] + Jh_psi[3] * Jpsi_g[3, 2]

    h_chi = Jh_psi[0]
    return h, Lie_f_h, Lie_g_h, d, v_r, v_h, h_chi, Jh_psi


@njit(cache=True, fastmath=True)
def compute_h_and_lie_numba(
    translation_bt: np.ndarray,
    obs_pos: np.ndarray,
    vel_lineare: np.ndarray,
    v_obs: np.ndarray,
    Tr: float,
    a_s: float,
    C: float,
    obs_acc: np.ndarray,
    atol: float,
    tc: float = 0.0,
):
    """
    Compute barrier margin h, its drift Lie derivative L_f h, and its control Lie derivative L_g h.
    """
    h, Lie_f_h, Lie_g_h, d, v_r, v_h, h_chi, _ = compute_h_lie_and_grad_numba(
        translation_bt, obs_pos, vel_lineare, v_obs, Tr, a_s, C, obs_acc, atol, tc
    )
    return h, Lie_f_h, Lie_g_h, d, v_r, v_h, h_chi


# ------------------------------------------------------------
# 7) Full Joint-Space Constraint Assembly
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def compute_h_and_constraints_numba(
    translation_bt: np.ndarray,
    obs_pos: np.ndarray,
    vel_lineare: np.ndarray,
    v_obs: np.ndarray,
    Tr: float,
    a_s: float,
    C: float,
    obs_acc: np.ndarray,
    atol: float,
    Jlin: np.ndarray,
    dJlin: np.ndarray,
    dq: np.ndarray,
    gamma: float,
    HAS_CBF: bool,
    delta_H: float = 0.0,
    tc: float = 0.0,
    sigma_psi: np.ndarray = np.zeros(4, dtype=np.float64),
    beta: float = 3.0,
):
    """
    Assemble the linear inequality row and lower bound for the QP joint-acceleration optimization.

    Mathematical Mapping:
        The Cartesian acceleration mu = dot{v}_r is related to joint acceleration u = ddot{q} via:
            mu = J_lin * u + dJ_lin * dq
        Substituting into the Robust Control Barrier Function inequality:
            L_f h + L_g h * (J_lin * u + dJ_lin * dq) >= -gamma * (h - margin)
        Rearranging into standard QP constraint form (A_cbf * u >= bound):
            (L_g h * J_lin) * u >= - (L_g h * dJ_lin * dq) - L_f h - gamma * (h - margin)

    Parameters:
        translation_bt: Monitored robot frame position [m] (3,)
        obs_pos: Obstacle / keypoint position [m] (3,)
        vel_lineare: Robot linear velocity [m/s] (3,)
        v_obs: Obstacle velocity [m/s] (3,)
        Tr: Reaction time [s]
        a_s: Braking capability [m/s^2]
        C: Safety separation clearance [m]
        obs_acc: Obstacle acceleration [m/s^2] (3,)
        atol: Comparison tolerance
        Jlin: Translational Jacobian of the monitored robot frame (3, nq)
        dJlin: Time derivative of translational Jacobian (3, nq)
        dq: Current robot joint velocities [rad/s] (nq,)
        gamma: Barrier filter responsiveness gain [1/s]
        HAS_CBF: Boolean flag enabling barrier enforcement
        delta_H: Scalar uncertainty margin [m]
        tc: Sampling period [s]
        sigma_psi: Standard deviations [sigma_d, sigma_vr, sigma_vh, sigma_ah] for ellipsoidal uncertainty
        beta: Confidence scaling factor (e.g. 3.0 for 99.7% Gaussian confidence)

    Returns:
        h: Barrier margin [m]
        constraint_row: Gradient row vector w.r.t joint accelerations in R^(nq)
        constraint_bound: Scalar lower bound for QP inequality constraint
        d: Current distance [m]
        vr, vh: Projected velocities along line of sight [m/s]
    """
    h, Lie_f_h, Lie_g_h, d, vr, vh, h_chi, Jh_psi = compute_h_lie_and_grad_numba(
        translation_bt, obs_pos, vel_lineare, v_obs, Tr, a_s, C, obs_acc, atol, tc
    )

    n = Jlin.shape[1]
    constraint_row = np.zeros(n, dtype=np.float64)
    constraint_bound = 0.0

    if HAS_CBF:
        # Compute row vector: constraint_row = Lie_g_h @ Jlin
        for j in range(n):
            constraint_row[j] = (
                Lie_g_h[0] * Jlin[0, j] +
                Lie_g_h[1] * Jlin[1, j] +
                Lie_g_h[2] * Jlin[2, j]
            )

        # Compute apparent acceleration contribution: Lie_g_h @ (dJlin @ dq)
        tmp = np.zeros(n, dtype=np.float64)
        for j in range(n):
            tmp[j] = (
                Lie_g_h[0] * dJlin[0, j] +
                Lie_g_h[1] * dJlin[1, j] +
                Lie_g_h[2] * dJlin[2, j]
            )

        lg_dJ_dq = 0.0
        for j in range(n):
            lg_dJ_dq += tmp[j] * dq[j]

        # Calculate robust margin:
        # If full covariance/standard deviations are provided, use state-dependent quadratic form
        # sigma_eff = sqrt(grad_psi_h * Sigma_psi * grad_psi_h^T)
        has_covariance = (sigma_psi[0] > 0.0 or sigma_psi[1] > 0.0 or sigma_psi[2] > 0.0 or sigma_psi[3] > 0.0)
        if has_covariance:
            sigma_eff = np.sqrt(
                (Jh_psi[0] * sigma_psi[0]) ** 2 +
                (Jh_psi[1] * sigma_psi[1]) ** 2 +
                (Jh_psi[2] * sigma_psi[2]) ** 2 +
                (Jh_psi[3] * sigma_psi[3]) ** 2
            )
            robust_margin = beta * sigma_eff
            # Correct robust CBF constraint: minus sign, dimensional consistency [m/s]
            constraint_bound = -lg_dJ_dq - Lie_f_h - gamma * (h - robust_margin)
        else:
            # Scalar uncertainty bound (backward-compatible)
            constraint_bound = -lg_dJ_dq - Lie_f_h - gamma * h - h_chi * delta_H

    return h, constraint_row, constraint_bound, d, vr, vh


# ------------------------------------------------------------
# 8) Volumetric Capsule Segment-to-Segment Distance (Ericson)
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def segment_to_segment_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    r1: float = 0.0,
    r2: float = 0.0,
    atol: float = 1e-12,
):
    """
    Compute closest distance and closest points between two 3D line segments (or capsules).

    Implements Christer Ericson's algorithm (Real-Time Collision Detection, Sec. 5.1.9).
    Segment 1: S1(s) = p1 + s * (p2 - p1),  s in [0, 1] with capsule radius r1
    Segment 2: S2(t) = q1 + t * (q2 - q1),  t in [0, 1] with capsule radius r2

    Parameters:
        p1, p2: Endpoints of Segment 1 (e.g. robot link frame origins) [m] (3,)
        q1, q2: Endpoints of Segment 2 (e.g. human limb keypoints or obstacle) [m] (3,)
        r1: Radius of Segment 1 capsule [m]
        r2: Radius of Segment 2 capsule [m]
        atol: Numerical zero tolerance

    Returns:
        d_surf: Signed surface distance d = d_centerline - (r1 + r2) [m]
        c1: Closest point on S1 centerline [m] (3,)
        c2: Closest point on S2 centerline [m] (3,)
        p_r_surf: Closest point on S1 capsule surface [m] (3,)
        p_h_surf: Closest point on S2 capsule surface [m] (3,)
        s: Parameter s in [0, 1] along segment 1
        t: Parameter t in [0, 1] along segment 2
    """
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    if a <= atol and e <= atol:
        s = 0.0
        t = 0.0
    elif a <= atol:
        s = 0.0
        t = min(max(f / e, 0.0), 1.0)
    elif e <= atol:
        t = 0.0
        c = float(np.dot(d1, r))
        s = min(max(-c / a, 0.0), 1.0)
    else:
        c = float(np.dot(d1, r))
        b = float(np.dot(d1, d2))
        denom = a * e - b * b
        if denom > atol:
            s = min(max((b * f - c * e) / denom, 0.0), 1.0)
        else:
            s = 0.0
        t = (b * s + f) / e
        if t < 0.0:
            t = 0.0
            s = min(max(-c / a, 0.0), 1.0)
        elif t > 1.0:
            t = 1.0
            s = min(max((b - c) / a, 0.0), 1.0)

    c1 = p1 + s * d1
    c2 = q1 + t * d2
    diff = c1 - c2
    d_center = float(np.sqrt(np.dot(diff, diff)))
    d_surf = d_center - (r1 + r2)

    if d_center > atol:
        u = diff / d_center
    else:
        u = np.array([1.0, 0.0, 0.0])

    p_r_surf = c1 - r1 * u
    p_h_surf = c2 + r2 * u
    return d_surf, c1, c2, p_r_surf, p_h_surf, s, t
