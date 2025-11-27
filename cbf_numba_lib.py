# cbf_numba_lib.py
#
# Numba-accelerated CBF / kinematic helper functions used by the UR10 main loop.

import numpy as np
from numba import njit

from numba_kernels import fill_vel_rows, fill_acc_rows, append_cbf_rows_loop

# Default constants (kept here so JITted functions have literal defaults)
C_DEFAULT = 0.25   # [m]
Tr_DEFAULT = 0.15  # [s]
A_S_DEFAULT = 2.5  # [m/s^2]


@njit(cache=True)
def compute_h(d, v, C=C_DEFAULT, Tr=Tr_DEFAULT, a_s=A_S_DEFAULT, v_h=0.0):
    """
    Inverse equation: minimum separation needed to permit speed |v|.
    This matches your original piecewise logic.
    """
    h = 0.0
    if v < 0.0:
        if v_h > 0:
            dmin = C + v_h * Tr - v * Tr + v_h * (-v / a_s) + 0.5 * v ** 2 / a_s
            h = d - dmin
        elif v_h < v:
            if d >= C:
                h = d - C
            else:
                h = d - C + (C - d) * Tr / C * v
        else:
            h = d - C + (v - v_h) * Tr - (v_h - v) ** 2 * 0.5 / a_s

    else:
        if v_h < 0:
            dmin = C
            coef = Tr
        else:
            dmin = C + v_h * Tr
            coef = Tr + v_h / a_s

        if d < dmin:
            h = coef * v
        else:
            h = (d - dmin) + coef * v

    return h


@njit(cache=True)
def range_state_derivative(v_lin: np.ndarray, v_human: np.ndarray):
    """
    Compute f(chi) and g(chi) in one function.

    Parameters:
    - v_lin:    (3,) numpy array
    - v_human:  (3,) numpy array

    Returns:
    - f:        (12,) numpy array
    - g:        (12, 3) numpy array
    """
    zero3 = np.zeros(3)
    zero3x3 = np.zeros((3, 3))
    I3 = np.eye(3)

    # f(chi) = [v_lin; v_human; 0; 0]
    f = np.concatenate((v_lin, v_human, zero3, zero3))

    # g(chi) = [0; 0; I; 0]
    g = np.vstack((zero3x3, zero3x3, I3, zero3x3))

    return f, g


@njit(cache=True)
def jacobian_psi(p_r, p_h, v_lin, v_human):
    """
    Jacobian of the range-related state psi wrt chi = [p_r, p_h, v_r, v_h].
    Matches your original algebra, just written in a numba-friendly style.
    """
    diff = p_r - p_h
    norm = np.sqrt((diff ** 2).sum())
    u_rh = (diff / norm).reshape(3, 1)
    zero = np.zeros((1, 3))
    P = np.eye(3) - u_rh @ u_rh.T

    vlinP = v_lin.reshape(1, 3) @ P
    vhP = v_human.reshape(1, 3) @ P

    row1 = np.hstack((u_rh.T, -u_rh.T, zero, zero))
    row2 = np.hstack((vlinP, -vlinP, u_rh.T, zero))
    row3 = np.hstack((vhP, -vhP, zero, u_rh.T))

    jacobian = np.vstack((row1, row2, row3))
    return jacobian


@njit(cache=True)
def jacobian_h(d, v, v_h=0.0, a_h=0.0,
               C=C_DEFAULT, Tr=Tr_DEFAULT, a_s=A_S_DEFAULT):
    """
    Numba-friendly Jacobian (∂h/∂d, ∂h/∂v, ∂h/∂v_h) for the same h as compute_h
    (for v<0, it uses the optimization-based model; for v>=0, the simple branch).
    """
    eps = 1e-12

    # -------- Case 1: v >= 0 (simple branch) --------
    if v >= 0.0:
        if d < C:
            dh_dd = 0.0
            dh_dv = Tr
            dh_dvh = 0.0
        else:
            dh_dd = 1.0
            dh_dv = Tr
            dh_dvh = 0.0
        return dh_dd, dh_dv, dh_dvh

    # -------- Case 2: v < 0 (true optimization over [0, t_stop]) --------
    t_stop = Tr - v / a_s

    # Up to 5 candidates: [0, Tr, t_stop, t_star, t_prime]
    max_candidates = 5
    times = np.empty(max_candidates)
    vals = np.empty(max_candidates)
    n = 0

    # Helper: d_total(t)
    def d_total_local(t):
        # d_r(t)
        if t <= Tr:
            d_r = v * t
        elif t <= t_stop:
            d_r = v * t + 0.5 * a_s * (t - Tr) * (t - Tr)
        else:
            d_r = v * Tr - 0.5 * (v * v) / a_s

        # d_h(t)
        d_h = v_h * t + 0.5 * a_h * t * t

        return d + d_r - d_h

    # Endpoints
    t = 0.0
    times[n] = t
    vals[n] = d_total_local(t)
    n += 1

    t = Tr
    times[n] = t
    vals[n] = d_total_local(t)
    n += 1

    t = t_stop
    times[n] = t
    vals[n] = d_total_local(t)
    n += 1

    # Interior pre-Tr stationary point t* (if any)
    if np.abs(a_h) > eps:
        t_star = (v - v_h) / a_h
        if 0.0 < t_star < Tr:
            t = t_star
            times[n] = t
            vals[n] = d_total_local(t)
            n += 1

    # Interior post-Tr stationary point t' (if any)
    if np.abs(a_h - a_s) > eps:
        t_prime = ((v - v_h) - a_s * Tr) / (a_h - a_s)
        if Tr < t_prime < t_stop:
            t = t_prime
            times[n] = t
            vals[n] = d_total_local(t)
            n += 1

    # Pick minimizer
    idx_min = 0
    min_val = vals[0]
    for i in range(1, n):
        if vals[i] < min_val:
            min_val = vals[i]
            idx_min = i
    t_min = times[idx_min]

    # Derivatives as in your previous reasoning
    if np.abs(t_min - 0.0) <= 1e-9:
        dh_dd = 1.0
        dh_dv = 0.0
        dh_dvh = 0.0

    elif np.abs(t_min - Tr) <= 1e-9:
        dh_dd = 1.0
        dh_dv = Tr
        dh_dvh = -Tr

    elif np.abs(t_min - t_stop) <= 1e-9:
        vh_at_stop = v_h + a_h * t_stop
        dh_dd = 1.0
        dh_dv = t_stop + vh_at_stop / a_s
        dh_dvh = -t_stop

    else:
        # interior minima
        dh_dd = 1.0
        dh_dv = t_min
        dh_dvh = -t_min

    return dh_dd, dh_dv, dh_dvh


def damped_pinv_svd(J, lam=1e-4):
    """
    Damped pseudo-inverse via SVD.
    Left non-jitted since it's called only in the non-CBF branch and uses SVD.
    """
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_damped = S / (S ** 2 + lam ** 2)  # approx S^-1
    return (Vt.T * S_damped) @ U.T

# ------------------------------------------------------------
# High-level: assemble ALL constraints, and objective parts
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def assemble_qp_PID_problem(
    # outputs (in-place)
    A, c,
    # inputs
    FreeVel, ForcedVel,
    q, dq,
    Dq_max, DDq_max,
    # CBF inputs (optional; pass empty arrays if unused)
    frames_p, frames_vlin, Jlins, dJlins, obs_p, obs_v, obs_a,
    Tr, a_s, C, gamma, atol
):
    nq = q.size
    # zero A, c
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = 0.0
        c[i] = 0.0

    # constraint rows
    row = 0
    # overwrite last line's rhs to -DDtraj_max using c[row_idx] once caller fixes it if needed

    x0 = np.empty(nq * 2)
    for i in range(nq):
        x0[i] = q[i]
        x0[nq + i] = dq[i]

    # row = fill_tube_rows(A, c, row, nq, FreePos, ForcedPos, x0, nominal_q, delta_q_max)
    row = fill_vel_rows(A, c, row, nq, FreeVel, ForcedVel, x0, Dq_max)
    row = fill_acc_rows(A, c, row, nq, DDq_max)

    # CBF rows (if any)
    if frames_p.size != 0 and obs_p.size != 0:
        row, hmin, dmin, vrel_min = append_cbf_rows_loop(
            A, c, row, frames_p, frames_vlin, obs_p, obs_v, obs_a, Jlins, dJlins, dq, Tr, a_s, C, gamma, atol
        )
    else:
        hmin = 1e9
        dmin = 1e9
        vrel_min = 1e9  # unused placeholder
    # objective parts

    return row, hmin, dmin, vrel_min
