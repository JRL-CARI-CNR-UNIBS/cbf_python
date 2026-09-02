# numba_kernels.py
"""
Numba-accelerated QP matrices and constraint assembly for BCFOptimalController.

This module provides in-place JIT-compiled kernels to construct:
- 1-step discrete state predictor matrices (FreePos, ForcedPos, FreeVel, ForcedVel).
- Scaling bounds and acceleration limits on trajectory scaling tau_ddot.
- Bounding tube constraints on joint configuration error delta_q.
- Joint velocity and acceleration limits.
- Speed and Separation Monitoring (SSM) Control Barrier Function constraint rows.
- Quadratic and linear objective components (P, b) for position tracking, velocity scaling,
  and time-scaling regularization.
"""

import numpy as np
from numba import njit

from Controller.Numba_scripts.ssm_cbf_acc import compute_h_and_constraints_numba


# ---------------------------------------------------------------------------
# 1) Discrete State Prediction Matrices
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def build_free_forced_one_step(Ts: float, nq: int):
    """
    Constructs discrete 1-step integration matrices for double integrator dynamics:
        q(k+1) = FreePos * [q(k), dq(k)]^T + ForcedPos * ddq(k)
        dq(k+1) = FreeVel * [q(k), dq(k)]^T + ForcedVel * ddq(k)
    """
    I = np.eye(nq)
    ForcedPos = 0.5 * (Ts ** 2) * I
    FreePos = np.hstack((I, Ts * I))
    ForcedVel = Ts * I
    FreeVel = np.hstack((np.zeros_like(I), I))
    return FreePos, ForcedPos, FreeVel, FreeVel


# ---------------------------------------------------------------------------
# 2) Constraint Row Fillers (In-Place)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def fill_scaling_rows(A, c, row, nq, Tc, Dtraj, DDtraj_max):
    """
    Enforces bounds on trajectory scaling rate s_dot and its acceleration s_ddot:
        - 0 <= s_dot(k+1) <= 1  -->  -Tc * s_ddot <= 1 - s_dot,  Tc * s_ddot <= s_dot
        - s_ddot >= -DDtraj_max  -->  -s_ddot <= DDtraj_max
    """
    # Upper bound: s_dot + Tc * s_ddot <= 1
    for j in range(nq):
        A[row, j] = 0.0
    A[row, nq] = -Tc
    c[row] = -(1.0 - Dtraj)
    row += 1

    # Lower bound: s_dot + Tc * s_ddot >= 0
    for j in range(nq):
        A[row, j] = 0.0
    A[row, nq] = +Tc
    c[row] = -Dtraj
    row += 1

    # Maximum deceleration: s_ddot >= -DDtraj_max
    for j in range(nq):
        A[row, j] = 0.0
    A[row, nq] = -1.0
    c[row] = -DDtraj_max
    row += 1
    return row


@njit(cache=True, fastmath=True)
def fill_tube_rows(A, c, row, nq, FreePos, ForcedPos, x0, nominal_q, delta_q_max):
    """
    Enforces bounding tube constraints around nominal trajectory:
        | q(k+1) - q_nom(k+1) | <= delta_q_max
    """
    Fx = FreePos @ x0

    # Lower bound: -ForcedPos * ddq <= -nominal_q - delta_q_max + Fx
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = -ForcedPos[i, j]
        c[row + i] = -nominal_q[i] - delta_q_max[i] + Fx[i]
    row += nq

    # Upper bound: +ForcedPos * ddq <= +nominal_q - delta_q_max - Fx
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = +ForcedPos[i, j]
        c[row + i] = +nominal_q[i] - delta_q_max[i] - Fx[i]
    row += nq
    return row


@njit(cache=True, fastmath=True)
def fill_vel_rows(A, c, row, nq, FreeVel, ForcedVel, x0, Dq_max):
    """
    Enforces joint velocity limits:
        | dq(k+1) | <= Dq_max
    """
    Fx = FreeVel @ x0

    # Lower limit: -ForcedVel * ddq <= -Dq_max + Fx
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = -ForcedVel[i, j]
        c[row + i] = -Dq_max[i] + Fx[i]
    row += nq

    # Upper limit: +ForcedVel * ddq <= -Dq_max - Fx
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = +ForcedVel[i, j]
        c[row + i] = -Dq_max[i] - Fx[i]
    row += nq
    return row


@njit(cache=True, fastmath=True)
def fill_acc_rows(A, c, row, nq, DDq_max):
    """
    Enforces joint acceleration limits:
        | ddq | <= DDq_max
    """
    # Lower limit: -ddq <= -DDq_max
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = -1.0 if i == j else 0.0
        c[row + i] = -DDq_max[i]
    row += nq

    # Upper limit: +ddq <= -DDq_max
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = +1.0 if i == j else 0.0
        c[row + i] = -DDq_max[i]
    row += nq
    return row


@njit(cache=True, fastmath=True)
def append_cbf_rows_loop(
    A, c, row,
    frames_p, frames_vlin,
    obs_p, obs_v, obs_a,
    Jlins, dJlins, dq,
    Tr, a_s, C, gamma, atol, HAS_CBF, keypoint_to_log
):
    """
    Appends SSM-CBF inequality constraints across all monitored robot frames and human keypoints.
    """
    hmin = 1e9
    htest = 1e9
    dmin = 1e9
    dtest = 1e9
    i_h = 190
    i_d = 190

    vr_min = 0.0
    vh_min = 0.0

    nF = frames_p.shape[0]
    nO = obs_p.shape[0]
    nq = dq.size

    for f in range(nF):
        p_bt = frames_p[f]
        vlin = frames_vlin[f]
        Jlin = Jlins[f]
        dJlin = dJlins[f]
        for o in range(nO):
            op = obs_p[o]
            ov = obs_v[o]
            oa = obs_a[o]
            h, row_vec, bound, d, vr, vh = compute_h_and_constraints_numba(
                p_bt, op, vlin, ov, Tr, a_s, C, oa, atol, Jlin, dJlin, dq, gamma, HAS_CBF
            )
            if keypoint_to_log >= 0:
                if o == min(keypoint_to_log, nO - 1) and f == frames_p.shape[0] - 1:
                    vr_min = vr
                    vh_min = vh
                    dmin = d
                    hmin = h
            else:
                if h < hmin:
                    vr_min = vr
                    vh_min = vh
                    dmin = d
                    hmin = h

            if HAS_CBF:
                for j in range(nq):
                    A[row, j] = row_vec[j]
                c[row] = bound
                row += 1

    return row, hmin, dmin, vr_min, vh_min, htest, dtest, i_h, i_d


# ---------------------------------------------------------------------------
# 3) Objective Assembly
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def assemble_objective_parts_inplace(
    P2, b_pos, b_vel, b_scaling,
    q, dq,
    nominal_q, nominal_Dq,
    Dtraj, Tc, ref_scaling
):
    """
    Assembles quadratic matrix components and linear gradient vectors for:
    - Position tracking: J_pos = 0.5 * || q(k+1) - q_nom(k+1) ||^2
    - Velocity scaling tracking: J_vel = 0.5 * || dq(k+1) - s_dot(k+1) * dq_nom(k+1) ||^2
    - Scaling regularization: J_scaling = 0.5 * (s_dot(k+1) - ref_scaling)^2
    """
    nq = q.size

    # Reset buffers
    for i in range(nq + 1):
        for j in range(nq + 1):
            P2[i, j] = 0.0
        b_pos[i] = 0.0
        b_vel[i] = 0.0
        b_scaling[i] = 0.0

    # Cross-coupling block P2 for velocity-scaling objective
    ndq_dot = 0.0
    for i in range(nq):
        val = -(Tc * Tc) * nominal_Dq[i]
        P2[i, nq] = val
        P2[nq, i] = val
        P2[i, i] = Tc ** 2
        ndq_dot += nominal_Dq[i] * nominal_Dq[i]
    P2[nq, nq] = (Tc * Tc) * ndq_dot

    # Linear position gradient b_pos
    half_T2 = 0.5 * Tc * Tc
    for i in range(nq):
        b_pos[i] = (nominal_q[i] - q[i] - dq[i] * Tc) * half_T2

    # Linear velocity gradient b_vel
    tmp_dot = 0.0
    for i in range(nq):
        val = (nominal_Dq[i] * Dtraj - dq[i]) * Tc
        b_vel[i] = val
        tmp_dot += (nominal_Dq[i] * Dtraj - dq[i]) * (nominal_Dq[i] * Tc)
    b_vel[nq] = -tmp_dot

    # Linear scaling gradient b_scaling
    b_scaling[nq] = -Tc * (Dtraj - ref_scaling)


# ---------------------------------------------------------------------------
# 4) High-Level In-Place QP Assembler
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def assemble_qp_inplace(
    P2, b_pos, b_vel, b_scaling,
    A, c,
    FreePos, ForcedPos, FreeVel, ForcedVel,
    q, dq,
    nominal_q, nominal_Dq,
    Dtraj, Tc,
    Dq_max, DDq_max, delta_q_max,
    frames_p, frames_vlin, Jlins, dJlins, obs_p, obs_v, obs_a,
    Tr, a_s, C, gamma, DDtraj_max, atol, ref_scaling, HAS_CBF, keypoint_to_log
):
    """
    Assembles complete QP objective and linear inequalities in-place for fast real-time execution.
    """
    nq = q.size
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = 0.0
        c[i] = 0.0

    # Constraint blocks
    row = 0
    row = fill_scaling_rows(A, c, row, nq, Tc, Dtraj, DDtraj_max=DDtraj_max)

    x0 = np.empty(nq * 2)
    for i in range(nq):
        x0[i] = q[i]
        x0[nq + i] = dq[i]

    row = fill_tube_rows(A, c, row, nq, FreePos, ForcedPos, x0, nominal_q, delta_q_max)
    row = fill_vel_rows(A, c, row, nq, FreeVel, ForcedVel, x0, Dq_max)
    row = fill_acc_rows(A, c, row, nq, DDq_max)

    # CBF constraint block
    if frames_p.size != 0 and obs_p.size != 0:
        row, hmin, dmin, vr_min, vh_min, htest, dtest, i_h, i_d = append_cbf_rows_loop(
            A, c, row, frames_p, frames_vlin, obs_p, obs_v, obs_a, Jlins, dJlins, dq, Tr, a_s, C, gamma, atol, HAS_CBF,
            keypoint_to_log
        )
    else:
        hmin = 1e9
        dmin = 1e9
        vr_min = 1e9
        vh_min = 1e9
        htest = 1e9
        dtest = 1e9
        i_h = 0
        i_d = 0

    # Objective blocks
    assemble_objective_parts_inplace(P2, b_pos, b_vel, b_scaling, q, dq, nominal_q, nominal_Dq, Dtraj, Tc, ref_scaling)

    return row, hmin, dmin, vr_min, vh_min, htest, dtest, i_h, i_d