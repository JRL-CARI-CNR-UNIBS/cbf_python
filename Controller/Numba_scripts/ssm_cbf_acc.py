# ssm_cbf_acc.py
"""
Analytical Speed and Separation Monitoring (SSM) Control Barrier Functions (CBFs)
with Numba acceleration.

This module provides analytical formulations and JIT-compiled kernels for:
1. State derivative construction for range/separation dynamics.
2. Analytical minimum separation distance d_min and its gradients over candidate collision times.
3. Control Barrier Function h(x) with category-2 safety margins.
4. Fast contractions of the geometric Jacobian J_psi with system dynamics.
5. Lie derivatives (L_f h, L_g h) along robot and human trajectory predictions.
6. Analytical constraint row and bound generation for Quadratic Programming (QP).
"""

import numpy as np
from numba import njit, float64, boolean


# ---------------------------------------------------------------------------
# 1) Dynamics: f(x) and g(x)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def range_state_derivative_numba(v_r: np.ndarray, v_h: np.ndarray):
    """
    Constructs the state drift f(x) and input matrix g(x) for relative range dynamics.

    Parameters:
        v_r : np.ndarray of shape (3,) - Robot Cartesian linear velocity.
        v_h : np.ndarray of shape (3,) - Human Cartesian linear velocity.

    Returns:
        f : np.ndarray of shape (12,) - State drift vector.
        g : np.ndarray of shape (12, 3) - Input mapping matrix for robot acceleration.
    """
    f = np.zeros(12, dtype=np.float64)
    f[0:3] = v_r
    f[3:6] = v_h

    g = np.zeros((12, 3), dtype=np.float64)
    g[6, 0] = 1.0
    g[7, 1] = 1.0
    g[8, 2] = 1.0
    return f, g


# ---------------------------------------------------------------------------
# 2) Analytical Minimum Distance & Jacobian
# ---------------------------------------------------------------------------
@njit((float64, float64, float64, float64, float64, float64, float64), cache=True, fastmath=True)
def dmin_and_jacobian_numba(d, v_r, v_h, a_h, tr, a_max, atol):
    """
    Evaluates the analytical minimum separation distance d_min and its gradient
    with respect to [d, v_r, v_h, a_h] across candidate critical instants [t0, t1, t2, t3, t4].

    Parameters:
        d     : float - Current Euclidean distance between robot link and human keypoint.
        v_r   : float - Projected relative velocity of robot along line of sight.
        v_h   : float - Projected relative velocity of human along line of sight.
        a_h   : float - Projected acceleration of human along line of sight.
        tr    : float - System reaction time [s].
        a_max : float - Maximum robot braking deceleration (a_s > 0) [m/s^2].
        atol  : float - Numerical tolerance for candidate time singularity checks.

    Returns:
        d_min : float - Analytical worst-case minimum distance.
        jac   : np.ndarray of shape (4,) - Gradient [dd_min/dd, dd_min/dv_r, dd_min/dv_h, dd_min/da_h].
    """
    if a_max <= 0.0:
        raise ValueError("a_max must be positive.")

    t0 = 0.0
    t2 = tr
    t_dec = -v_r / a_max
    if t_dec < 0.0:
        t_dec = 0.0
    t4 = t2 + t_dec
    m = a_h - a_max
    v_diff = v_r - v_h

    # Stationary points in intervals [t0, t2] and [t2, t4]
    has_t1 = False
    t1 = 0.0
    if abs(a_h) > atol:
        t1_raw = v_diff / a_h
        if (t0 + atol) < t1_raw < (t2 - atol):
            t1 = t1_raw
            has_t1 = True

    has_t3 = False
    t3 = 0.0
    if abs(m) > atol and (t4 - t2) > atol:
        t3_raw = (v_diff - a_max * t2) / m
        if (t2 + atol) < t3_raw < (t4 - atol):
            t3 = t3_raw
            has_t3 = True

    C1 = has_t1
    C3 = has_t3

    # Fixed-size candidate buffer
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
        add_candidate(t0)
        add_candidate(t4)
        if C1:
            add_candidate(t1)
        if C3:
            add_candidate(t3)
    else:
        add_candidate(t0)
        add_candidate(t2)
        if C1:
            add_candidate(t1)

    # Evaluate distance trajectory at candidate instants
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        tt = uniq[i]
        if tt <= t2 + atol:
            vals[i] = d + v_diff * tt - 0.5 * a_h * (tt * tt)
        else:
            vals[i] = d + v_diff * tt - 0.5 * a_h * (tt * tt) + 0.5 * a_max * (tt - t2) * (tt - t2)

    # Find minimum
    i_min = 0
    best = vals[0]
    for i in range(1, n):
        if vals[i] < best:
            best = vals[i]
            i_min = i

    t_star = uniq[i_min]
    d_min = best

    # Gradient computation at active candidate t_star: [d, v_r, v_h, a_h]
    jac = np.zeros(4, dtype=np.float64)

    if abs(t_star - t0) <= atol:
        jac[0] = 1.0
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
        jac[3] = -0.5 * t2 * t2
        return d_min, jac

    if has_t3 and abs(t_star - t3) <= atol:
        jac[0] = 1.0
        jac[1] = t3
        jac[2] = -t3
        jac[3] = -0.5 * (t3 * t3)
        return d_min, jac

    # t4 boundary
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


# ---------------------------------------------------------------------------
# 3) Barrier Function & Corrective Terms
# ---------------------------------------------------------------------------
@njit((float64, float64, float64, float64, float64, float64, float64, float64), cache=True, fastmath=True)
def h_and_jacobian_numba(d, v_r, v_h, a_h, tr, a_max, C, atol):
    """
    Evaluates safety barrier function h and its gradient, incorporating safety clearance C
    and Category-2 proximity correction when d_min < C.
    """
    d_min, dist_jac = dmin_and_jacobian_numba(d, v_r, v_h, a_h, tr, a_max, atol)

    h_val = d_min - C
    h_jac = np.empty(4, dtype=np.float64)
    h_jac[0] = dist_jac[0]
    h_jac[1] = dist_jac[1]
    h_jac[2] = dist_jac[2]
    h_jac[3] = dist_jac[3]

    if d_min < C:
        factor = (tr / C) * v_r
        h_val += ((C - d_min) / C) * tr * v_r
        h_jac[0] += -factor * dist_jac[0]
        h_jac[1] += -factor * dist_jac[1] + ((C - d_min) * tr / C)
        h_jac[2] += -factor * dist_jac[2]
        h_jac[3] += -factor * dist_jac[3]

    return h_val, h_jac


# ---------------------------------------------------------------------------
# 4) Geometric Jacobians
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def jacobian_psi_numba(p_r: np.ndarray, p_h: np.ndarray, v_lin: np.ndarray, v_human: np.ndarray):
    """
    Evaluates the 4x12 Jacobian of the measurement map psi = [d, v_r, v_h, a_h]^T.
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


# ---------------------------------------------------------------------------
# 5) Fast Contractions: (Jpsi @ f) and (Jpsi @ g)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def jacobian_psi_times_fg_fast_numba(
    p_r: np.ndarray,
    p_h: np.ndarray,
    v_r: np.ndarray,
    v_h: np.ndarray,
    atol: float = 1e-12,
):
    """
    Direct, allocation-free evaluation of J_psi @ f(x) and J_psi @ g(x).

    Returns:
        Jpsi_f : np.ndarray of shape (4,)
        Jpsi_g : np.ndarray of shape (4, 3)
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

    # Positional gradient vectors (tangential velocity / distance)
    grad_pr_vr = (v_r - u * vr_rel) / d
    grad_pr_vh = (v_h - u * vh_rel) / d

    Jpsi_f = np.zeros(4, dtype=np.float64)
    Jpsi_f[0] = u[0] * v_diff[0] + u[1] * v_diff[1] + u[2] * v_diff[2]
    Jpsi_f[1] = grad_pr_vr[0] * v_diff[0] + grad_pr_vr[1] * v_diff[1] + grad_pr_vr[2] * v_diff[2]
    Jpsi_f[2] = grad_pr_vh[0] * v_diff[0] + grad_pr_vh[1] * v_diff[1] + grad_pr_vh[2] * v_diff[2]

    Jpsi_g = np.zeros((4, 3), dtype=np.float64)
    Jpsi_g[1, 0] = u[0]
    Jpsi_g[1, 1] = u[1]
    Jpsi_g[1, 2] = u[2]

    return Jpsi_f, Jpsi_g


# ---------------------------------------------------------------------------
# 6) Lie Derivatives
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
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
    Computes g matrix, scalar Lie derivative L_f h, and 3D vector Lie derivative L_g h.
    """
    r0 = translation_bt[0] - obs_pos[0]
    r1 = translation_bt[1] - obs_pos[1]
    r2 = translation_bt[2] - obs_pos[2]
    d2 = r0 * r0 + r1 * r1 + r2 * r2

    if d2 <= atol * atol:
        nrm2 = vel_lineare[0] * vel_lineare[0] + vel_lineare[1] * vel_lineare[1] + vel_lineare[2] * vel_lineare[2]
        if nrm2 <= atol * atol:
            u0, u1, u2 = 1.0, 0.0, 0.0
        else:
            inv = 1.0 / np.sqrt(nrm2)
            u0 = vel_lineare[0] * inv
            u1 = vel_lineare[1] * inv
            u2 = vel_lineare[2] * inv
        d = np.sqrt(d2) if d2 > 0.0 else atol
    else:
        inv = 1.0 / np.sqrt(d2)
        u0, u1, u2 = r0 * inv, r1 * inv, r2 * inv
        d = 1.0 / inv

    v_rel = u0 * vel_lineare[0] + u1 * vel_lineare[1] + u2 * vel_lineare[2]
    v_h = u0 * v_obs[0] + u1 * v_obs[1] + u2 * v_obs[2]

    h_val, Jh_psi = h_and_jacobian_numba(d, v_rel, v_h, a_h, Tr, a_s, C, atol)

    Jpsi_f, Jpsi_g = jacobian_psi_times_fg_fast_numba(
        p_r=translation_bt, p_h=obs_pos, v_r=vel_lineare, v_h=v_obs
    )

    Lie_f_h = Jh_psi[0] * Jpsi_f[0] + Jh_psi[1] * Jpsi_f[1] + Jh_psi[2] * Jpsi_f[2] + Jh_psi[3] * Jpsi_f[3]

    Lie_g_h = np.zeros(3, dtype=np.float64)
    Lie_g_h[0] = Jh_psi[0] * Jpsi_g[0, 0] + Jh_psi[1] * Jpsi_g[1, 0] + Jh_psi[2] * Jpsi_g[2, 0] + Jh_psi[3] * Jpsi_g[3, 0]
    Lie_g_h[1] = Jh_psi[0] * Jpsi_g[0, 1] + Jh_psi[1] * Jpsi_g[1, 1] + Jh_psi[2] * Jpsi_g[2, 1] + Jh_psi[3] * Jpsi_g[3, 1]
    Lie_g_h[2] = Jh_psi[0] * Jpsi_g[0, 2] + Jh_psi[1] * Jpsi_g[1, 2] + Jh_psi[2] * Jpsi_g[2, 2] + Jh_psi[3] * Jpsi_g[3, 2]

    g = np.zeros((12, 3), dtype=np.float64)
    g[6, 0] = 1.0
    g[7, 1] = 1.0
    g[8, 2] = 1.0

    return g, Lie_f_h, Lie_g_h


@njit(
    (
        float64[:],  # translation_bt (3,)
        float64[:],  # obs_pos       (3,)
        float64[:],  # linear_vel   (3,)
        float64[:],  # v_obs         (3,)
        float64,     # Tr
        float64,     # a_s
        float64,     # C
        float64[:],  # obs_acc       (3,)
        float64,     # atol
    ),
    cache=True,
    fastmath=True,
)
def compute_h_and_lie_numba(translation_bt, obs_pos, vel_lineare, v_obs, Tr, a_s, C, obs_acc, atol):
    """
    Computes barrier function h and Lie derivatives L_f h and L_g h for a link-human pair.

    Returns:
        h       : float - Barrier value.
        Lie_f_h : float - Drift Lie derivative.
        Lie_g_h : np.ndarray of shape (3,) - Control Lie derivative vector.
        d       : float - Instantaneous distance.
        v_r     : float - Robot approach velocity along line of sight.
        v_h     : float - Human approach velocity along line of sight.
    """
    r0 = translation_bt[0] - obs_pos[0]
    r1 = translation_bt[1] - obs_pos[1]
    r2 = translation_bt[2] - obs_pos[2]
    d = np.sqrt(r0 * r0 + r1 * r1 + r2 * r2)

    if d <= atol:
        u0, u1, u2 = 1.0, 0.0, 0.0
        if d < atol:
            d = atol
    else:
        invd = 1.0 / d
        u0, u1, u2 = r0 * invd, r1 * invd, r2 * invd

    v_r = u0 * vel_lineare[0] + u1 * vel_lineare[1] + u2 * vel_lineare[2]
    v_h = u0 * v_obs[0] + u1 * v_obs[1] + u2 * v_obs[2]
    a_h = u0 * obs_acc[0] + u1 * obs_acc[1] + u2 * obs_acc[2]

    h, Jh_psi = h_and_jacobian_numba(d, v_r, v_h, a_h, Tr, a_s, C, atol)

    Jpsi_f, Jpsi_g = jacobian_psi_times_fg_fast_numba(translation_bt, obs_pos, vel_lineare, v_obs, atol)

    Lie_f_h = Jh_psi[0] * Jpsi_f[0] + Jh_psi[1] * Jpsi_f[1] + Jh_psi[2] * (Jpsi_f[2] + a_h) + Jh_psi[3] * Jpsi_f[3]

    Lie_g_h = np.zeros(3, dtype=np.float64)
    Lie_g_h[0] += Jh_psi[0] * Jpsi_g[0, 0] + Jh_psi[1] * Jpsi_g[1, 0] + Jh_psi[2] * Jpsi_g[2, 0] + Jh_psi[3] * Jpsi_g[3, 0]
    Lie_g_h[1] += Jh_psi[0] * Jpsi_g[0, 1] + Jh_psi[1] * Jpsi_g[1, 1] + Jh_psi[2] * Jpsi_g[2, 1] + Jh_psi[3] * Jpsi_g[3, 1]
    Lie_g_h[2] += Jh_psi[0] * Jpsi_g[0, 2] + Jh_psi[1] * Jpsi_g[1, 2] + Jh_psi[2] * Jpsi_g[2, 2] + Jh_psi[3] * Jpsi_g[3, 2]

    return h, Lie_f_h, Lie_g_h, d, v_r, v_h


# ---------------------------------------------------------------------------
# 7) Full Constraint Assembly
# ---------------------------------------------------------------------------
@njit(
    (
        float64[:],    # translation_bt (3,)
        float64[:],    # obs_pos       (3,)
        float64[:],    # linear_vel   (3,)
        float64[:],    # v_obs         (3,)
        float64,       # Tr
        float64,       # a_s
        float64,       # C
        float64[:],    # obs_acc       (3,)
        float64,       # atol
        float64[:, :], # Jlin  (3 x n)
        float64[:, :], # dJlin (3 x n)
        float64[:],    # dq    (n,)
        float64,       # gamma
        boolean,       # HAS_CBF
    ),
    cache=True,
    fastmath=True,
)
def compute_h_and_constraints_numba(
    translation_bt, obs_pos, vel_lineare, v_obs,
    Tr, a_s, C, obs_acc, atol, Jlin, dJlin, dq, gamma, HAS_CBF
):
    """
    Assembles the QP inequality constraint row and bound for a link-human obstacle pair:
        L_g h * J_lin * ddq >= - L_g h * dJ_lin * dq - L_f h - gamma * h
    Expressed in <= standard QP form:
        A_row * ddq <= c_bound
    where:
        A_row   = -(L_g h * J_lin)
        c_bound = L_g h * dJ_lin * dq + L_f h + gamma * h
    """
    h, Lie_f_h, Lie_g_h, d, vr, vh = compute_h_and_lie_numba(
        translation_bt, obs_pos, vel_lineare, v_obs, Tr, a_s, C, obs_acc, atol
    )

    n = Jlin.shape[1]
    constraint_row = np.zeros(n, dtype=np.float64)
    constraint_bound = 0.0

    if HAS_CBF:
        for j in range(n):
            constraint_row[j] = (
                Lie_g_h[0] * Jlin[0, j] +
                Lie_g_h[1] * Jlin[1, j] +
                Lie_g_h[2] * Jlin[2, j]
            )

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

        constraint_bound = -lg_dJ_dq - Lie_f_h - gamma * h

    return h, constraint_row, constraint_bound, d, vr, vh
