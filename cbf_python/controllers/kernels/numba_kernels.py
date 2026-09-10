"""
Numba accelerated QP assembly kernels and constraint row generators.

Optimized with @njit(cache=True, fastmath=True) for high-frequency control loops.
"""

from __future__ import annotations
import numpy as np
from numba import njit

from cbf_python.controllers.kernels.ssm_cbf_acc import compute_h_and_constraints_numba


# ------------------------------------------------------------
# 1) Primitive state transition blocks (Discrete integrator)
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def build_free_forced_one_step(Ts: float, nq: int):
    """Build discrete-time transition matrices for position and velocity under constant acceleration."""
    I = np.eye(nq)
    ForcedPos = 0.5 * (Ts ** 2) * I
    FreePos = np.hstack((I, Ts * I))
    ForcedVel = Ts * I
    FreeVel = np.hstack((np.zeros_like(I), I))
    return FreePos, ForcedPos, FreeVel, ForcedVel


# ------------------------------------------------------------
# 2) Constraint row builders (in-place assembly into A and c)
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def fill_scaling_rows(A, c, row, nq, Tc, Dtraj, DDtraj_max):
    """Assemble time-scaling constraints: Dtrajectory_time bounds and derivatives."""
    # [-0..0, -Tc]*[ddq, DDtraj] <= -(1 - Dtraj) -> Dtraj(k+1) <= 1
    for j in range(nq):
        A[row, j] = 0.0
    A[row, nq] = -Tc
    c[row] = -(1.0 - Dtraj)
    row += 1

    # [0..0, +Tc]*[...] <= -Dtraj -> Dtraj(k+1) >= 0
    for j in range(nq):
        A[row, j] = 0.0
    A[row, nq] = +Tc
    c[row] = -Dtraj
    row += 1

    # [0..0, -1]*[...] <= -DDtraj_max
    for j in range(nq):
        A[row, j] = 0.0
    A[row, nq] = -1.0
    c[row] = -DDtraj_max
    row += 1
    return row


@njit(cache=True, fastmath=True)
def fill_tube_rows(A, c, row, nq, FreePos, ForcedPos, x0, nominal_q, delta_q_max):
    """Assemble position error tube constraints: |q(k+1) - nominal_q(k+1)| <= delta_q_max."""
    Fx = FreePos @ x0  # Shape (nq,)
    # Lower bound: -ForcedPos * ddq <= -nominal_q - delta_q_max + Fx
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = -ForcedPos[i, j]
        c[row + i] = -nominal_q[i] - delta_q_max[i] + Fx[i]
    row += nq

    # Upper bound: +ForcedPos * ddq <= nominal_q - delta_q_max - Fx
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = +ForcedPos[i, j]
        c[row + i] = nominal_q[i] - delta_q_max[i] - Fx[i]
    row += nq
    return row


@njit(cache=True, fastmath=True)
def fill_vel_rows(A, c, row, nq, FreeVel, ForcedVel, x0, Dq_max):
    """Assemble joint velocity limit constraints: |dq(k+1)| <= Dq_max."""
    Fv = FreeVel @ x0  # Shape (nq,)
    # -ForcedVel * ddq <= -Dq_max + Fv
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = -ForcedVel[i, j]
        c[row + i] = -Dq_max[i] + Fv[i]
    row += nq

    # +ForcedVel * ddq <= -Dq_max - Fv
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = +ForcedVel[i, j]
        c[row + i] = -Dq_max[i] - Fv[i]
    row += nq
    return row


@njit(cache=True, fastmath=True)
def fill_acc_rows(A, c, row, nq, DDq_max):
    """Assemble joint acceleration limit constraints: |ddq| <= DDq_max."""
    # -I * ddq <= -DDq_max
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = -1.0 if i == j else 0.0
        c[row + i] = -DDq_max[i]
    row += nq

    # +I * ddq <= -DDq_max
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = +1.0 if i == j else 0.0
        c[row + i] = -DDq_max[i]
    row += nq
    return row


@njit(cache=True, fastmath=True)
def append_cbf_rows_loop(
    A, c, row,
    frames_p, frames_vlin,  # (nF, 3), (nF, 3)
    obs_p, obs_v, obs_a,    # (nO, 3)
    Jlins, dJlins, dq,      # (nF, 3, nq)
    Tr, a_s, C, gamma, atol, HAS_CBF, keypoint_to_log,
):
    """Loop over all monitored robot control points and obstacles to assemble CBF constraints."""
    hmin = 1e9
    htest = 1e9
    dmin = 1e9
    dtest = 1e9
    i_h = 0
    i_d = 0

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
            # If keypoint_to_log is non-negative, select that keypoint; otherwise select global minimum
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

            if d < dtest:
                dtest = d
                i_d = o
            if h < htest:
                htest = h
                i_h = o

            if HAS_CBF:
                for j in range(nq):
                    A[row, j] = row_vec[j]
                A[row, nq] = 0.0
                c[row] = bound
                row += 1

    return row, hmin, dmin, vr_min, vh_min, htest, dtest, i_h, i_d


# ------------------------------------------------------------
# 3) Objective assembly: assemble P and b in-place
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def assemble_objective_parts_inplace(
    P2, b_pos, b_vel, b_scaling,
    q, dq,
    nominal_q, nominal_Dq,
    Dtraj, Tc, ref_scaling,
):
    """Assemble cost function Hessian and gradient blocks."""
    nq = q.size
    for i in range(nq + 1):
        for j in range(nq + 1):
            P2[i, j] = 0.0
        b_pos[i] = 0.0
        b_vel[i] = 0.0
        b_scaling[i] = 0.0

    # P2: coupling between joint acceleration and time scaling acceleration
    ndq_dot = 0.0
    for i in range(nq):
        val = -(Tc * Tc) * nominal_Dq[i]
        P2[i, nq] = val
        P2[nq, i] = val
        P2[i, i] = Tc ** 2
        ndq_dot += nominal_Dq[i] * nominal_Dq[i]
    P2[nq, nq] = (Tc * Tc) * ndq_dot

    # b_pos: tracking nominal position
    half_T2 = 0.5 * Tc * Tc
    for i in range(nq):
        b_pos[i] = (nominal_q[i] - q[i] - dq[i] * Tc) * half_T2

    # b_vel: tracking nominal velocity scaled by Dtraj
    tmp_dot = 0.0
    for i in range(nq):
        val = (nominal_Dq[i] * Dtraj - dq[i]) * Tc
        b_vel[i] = val
        tmp_dot += (nominal_Dq[i] * Dtraj - dq[i]) * (nominal_Dq[i] * Tc)
    b_vel[nq] = -tmp_dot

    # b_scaling: penalize deviation from reference scaling
    b_scaling[nq] = -Tc * (Dtraj - ref_scaling)


# ------------------------------------------------------------
# 4) High-level: assemble ALL constraints and objective parts
# ------------------------------------------------------------
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
    Tr, a_s, C, gamma, DDtraj_max, atol, ref_scaling, HAS_CBF, keypoint_to_log,
):
    """Zero out arrays, assemble scaling/tube/velocity/acceleration/CBF constraints, and assemble cost."""
    nq = q.size
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = 0.0
        c[i] = 0.0

    row = 0
    row = fill_scaling_rows(A, c, row, nq, Tc, Dtraj, DDtraj_max=DDtraj_max)

    x0 = np.empty(nq * 2)
    for i in range(nq):
        x0[i] = q[i]
        x0[nq + i] = dq[i]

    row = fill_tube_rows(A, c, row, nq, FreePos, ForcedPos, x0, nominal_q, delta_q_max)
    row = fill_vel_rows(A, c, row, nq, FreeVel, ForcedVel, x0, Dq_max)
    row = fill_acc_rows(A, c, row, nq, DDq_max)

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

    assemble_objective_parts_inplace(P2, b_pos, b_vel, b_scaling, q, dq, nominal_q, nominal_Dq, Dtraj, Tc, ref_scaling)
    return row, hmin, dmin, vr_min, vh_min, htest, dtest, i_h, i_d
