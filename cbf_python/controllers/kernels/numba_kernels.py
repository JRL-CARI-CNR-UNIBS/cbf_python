"""
Numba-accelerated Quadratic Program (QP) assembly kernels and constraint generators.

This module provides high-performance, in-place Numba kernels compiled with
@njit(cache=True, fastmath=True) for high-frequency (500 Hz) robotic motion control.

Mathematical Overview:
    At each discrete control cycle k, the optimization problem determines the optimal
    joint acceleration u_k in R^nq and path-scaling acceleration ddot{xi}_k in R:
        min_{u_k, ddot{xi}_k, eps_k}
            lambda_pos * ||q_{k+1} - q_nom_{k+1}||^2
          + lambda_vel * ||dq_{k+1} - dq_nom_{k+1}||^2
          + lambda_acc * ||u_k||^2
          + lambda_scaling * (dot{xi}_{k+1} - dot{xi}_ref)^2
          + lambda_tube * ||eps_k||^2
    Subject to:
        1. Time-scaling bounds: 0 <= dot{xi}_{k+1} <= 1, ddot{xi}_k <= ddot{xi}_max
        2. Joint position limits: q_min <= q_{k+1} <= q_max
        3. Joint velocity limits: |dq_{k+1}| <= dq_max
        4. Joint acceleration limits: |u_k| <= ddq_max
        5. Joint-space trajectory tube: |q_{k+1} - q_nom_{k+1}| <= delta_q_max + eps_k, with eps_k >= 0
        6. Pairwise Robust Control Barrier Function (CBF) constraints: A_cbf * u_k >= bound
"""

from __future__ import annotations
import numpy as np
from numba import njit

from cbf_python.controllers.kernels.ssm_cbf_acc import (
    compute_h_and_constraints_numba,
    segment_to_segment_distance,
)


# ------------------------------------------------------------
# 1) Primitive State Transition Blocks (Discrete Integrator)
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def build_free_forced_one_step(Ts: float, nq: int):
    """
    Construct discrete-time state transition matrices under constant acceleration.

    Discrete integration over time step Ts:
        q(k+1)  = q(k) + Ts * dq(k) + 0.5 * Ts^2 * u(k)
                = FreePos * [q(k); dq(k)] + ForcedPos * u(k)
        dq(k+1) = dq(k) + Ts * u(k)
                = FreeVel * [q(k); dq(k)] + ForcedVel * u(k)

    Parameters:
        Ts: Control sampling period [s]
        nq: Number of robot degrees of freedom (joints)

    Returns:
        FreePos: Free position evolution matrix (nq, 2*nq)
        ForcedPos: Input-driven position matrix (nq, nq) = 0.5 * Ts^2 * I
        FreeVel: Free velocity evolution matrix (nq, 2*nq)
        ForcedVel: Input-driven velocity matrix (nq, nq) = Ts * I
    """
    I = np.eye(nq)
    ForcedPos = 0.5 * (Ts ** 2) * I
    FreePos = np.hstack((I, Ts * I))
    ForcedVel = Ts * I
    FreeVel = np.hstack((np.zeros_like(I), I))
    return FreePos, ForcedPos, FreeVel, ForcedVel


# ------------------------------------------------------------
# 2) Constraint Row Builders (In-Place Assembly into A and c)
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def fill_scaling_rows(A, c, row, nq, Tc, Dtraj, DDtraj_max):
    """
    Assemble linear inequality constraints for trajectory time-scaling.

    Enforces bounds on the path progression rate dot{xi} and its derivative ddot{xi}:
        1. dot{xi}(k+1) <= 1  <=>  -Tc * ddot{xi} >= -(1 - dot{xi})
        2. dot{xi}(k+1) >= 0  <=>  +Tc * ddot{xi} >= -dot{xi}
        3. ddot{xi} <= DDtraj_max  <=>  -1 * ddot{xi} >= -DDtraj_max

    Convention in quadprog: A_qp^T * x >= c_qp, where A has constraints as rows.
    """
    # Upper velocity bound: dot{xi}(k+1) <= 1
    for j in range(A.shape[1]):
        A[row, j] = 0.0
    A[row, nq] = -Tc
    c[row] = -(1.0 - Dtraj)
    row += 1

    # Lower velocity bound: dot{xi}(k+1) >= 0 (prevents backward execution)
    for j in range(A.shape[1]):
        A[row, j] = 0.0
    A[row, nq] = +Tc
    c[row] = -Dtraj
    row += 1

    # Maximum scaling acceleration bound: ddot{xi} <= DDtraj_max
    for j in range(A.shape[1]):
        A[row, j] = 0.0
    A[row, nq] = -1.0
    c[row] = -DDtraj_max
    row += 1
    return row


@njit(cache=True, fastmath=True)
def fill_pos_rows(A, c, row, nq, FreePos, ForcedPos, x0, q_min, q_max):
    """
    Assemble joint position limit constraints: q_min <= q(k+1) <= q_max.

    Substitution:
        q(k+1) = FreePos * x0 + ForcedPos * ddq
        Upper: -ForcedPos * ddq >= -q_max + FreePos * x0
        Lower: +ForcedPos * ddq >= +q_min - FreePos * x0
    """
    Fx = FreePos @ x0  # Free evolution prediction in R^nq

    # Upper joint limits
    for i in range(nq):
        for j in range(A.shape[1]):
            A[row + i, j] = 0.0
        for j in range(nq):
            A[row + i, j] = -ForcedPos[i, j]
        c[row + i] = -q_max[i] + Fx[i]
    row += nq

    # Lower joint limits
    for i in range(nq):
        for j in range(A.shape[1]):
            A[row + i, j] = 0.0
        for j in range(nq):
            A[row + i, j] = +ForcedPos[i, j]
        c[row + i] = q_min[i] - Fx[i]
    row += nq
    return row


@njit(cache=True, fastmath=True)
def fill_tube_rows(A, c, row, nq, FreePos, ForcedPos, x0, nominal_q, delta_q_max):
    """
    Assemble joint-space path tube constraints: |q(k+1) - nominal_q(k+1)| <= delta_q_max + eps.

    If decision vector contains slack variables eps in R^nq (columns nq+1 ... 2*nq),
    the tube is softened to guarantee continuous feasibility during evasive maneuvers.

    Inequalities:
        Lower bound: q(k+1) - nominal_q >= -delta_q_max - eps
            <=> ForcedPos * ddq + eps >= nominal_q - delta_q_max - FreePos * x0
            <=> -ForcedPos * ddq - eps <= -nominal_q + delta_q_max + FreePos * x0
        Upper bound: q(k+1) - nominal_q <= delta_q_max + eps
            <=> -ForcedPos * ddq + eps >= -nominal_q - delta_q_max + FreePos * x0
            <=> +ForcedPos * ddq - eps <= nominal_q + delta_q_max - FreePos * x0
    """
    Fx = FreePos @ x0  # Shape (nq,)
    has_slack = (A.shape[1] >= 2 * nq + 1)

    # Lower deviation bound
    for i in range(nq):
        for j in range(A.shape[1]):
            A[row + i, j] = 0.0
        for j in range(nq):
            A[row + i, j] = -ForcedPos[i, j]
        if has_slack:
            A[row + i, nq + 1 + i] = 1.0  # Slack entry
        c[row + i] = -nominal_q[i] - delta_q_max[i] + Fx[i]
    row += nq

    # Upper deviation bound
    for i in range(nq):
        for j in range(A.shape[1]):
            A[row + i, j] = 0.0
        for j in range(nq):
            A[row + i, j] = +ForcedPos[i, j]
        if has_slack:
            A[row + i, nq + 1 + i] = 1.0  # Slack entry
        c[row + i] = nominal_q[i] - delta_q_max[i] - Fx[i]
    row += nq

    # Slack non-negativity: eps_i >= 0
    if has_slack:
        for i in range(nq):
            for j in range(A.shape[1]):
                A[row + i, j] = 0.0
            A[row + i, nq + 1 + i] = 1.0
            c[row + i] = 0.0
        row += nq

    return row


@njit(cache=True, fastmath=True)
def fill_vel_rows(A, c, row, nq, FreeVel, ForcedVel, x0, Dq_max):
    """
    Assemble joint velocity limit constraints: |dq(k+1)| <= Dq_max.

    Substitution:
        dq(k+1) = FreeVel * x0 + ForcedVel * ddq
        Upper: -ForcedVel * ddq >= -Dq_max + FreeVel * x0
        Lower: +ForcedVel * ddq >= -Dq_max - FreeVel * x0
    """
    Fv = FreeVel @ x0  # Shape (nq,)
    for i in range(nq):
        for j in range(A.shape[1]):
            A[row + i, j] = 0.0
        for j in range(nq):
            A[row + i, j] = -ForcedVel[i, j]
        c[row + i] = -Dq_max[i] + Fv[i]
    row += nq

    for i in range(nq):
        for j in range(A.shape[1]):
            A[row + i, j] = 0.0
        for j in range(nq):
            A[row + i, j] = +ForcedVel[i, j]
        c[row + i] = -Dq_max[i] - Fv[i]
    row += nq
    return row


@njit(cache=True, fastmath=True)
def fill_acc_rows(A, c, row, nq, DDq_max):
    """
    Assemble joint acceleration limit constraints: |ddq| <= DDq_max.

    Upper: -I * ddq >= -DDq_max
    Lower: +I * ddq >= -DDq_max
    """
    for i in range(nq):
        for j in range(A.shape[1]):
            A[row + i, j] = 0.0
        A[row + i, i] = -1.0
        c[row + i] = -DDq_max[i]
    row += nq

    for i in range(nq):
        for j in range(A.shape[1]):
            A[row + i, j] = 0.0
        A[row + i, i] = +1.0
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
    delta_H=0.0,
    tc=0.0,
    sigma_psi=np.zeros(4, dtype=np.float64),
    beta=3.291,
    capsule_segments=np.empty((0, 2), dtype=np.int64),
    r_robot=0.0,
    r_human=0.0,
):
    """
    Iterate over monitored robot kinematic frames or cylindrical capsules and obstacles to assemble CBFs.

    Evaluates pairwise separation barriers and writes inequality rows:
        Lie_g_h * Jlin * ddq >= bound
    where:
        bound = - Lie_g_h * dJlin * dq - Lie_f_h - gamma * (h - beta * sigma_eff)
    """
    hmin = 1e9
    htest = 1e9
    dmin = 1e9
    dtest = 1e9
    i_h = 0
    i_d = 0

    vr_min = 0.0
    vh_min = 0.0

    nO = obs_p.shape[0]
    nq = dq.size

    use_caps = (capsule_segments.shape[0] > 0)

    if use_caps:
        nSeg = capsule_segments.shape[0]
        for s in range(nSeg):
            f1 = capsule_segments[s, 0]
            f2 = capsule_segments[s, 1]
            p1 = frames_p[f1]
            p2 = frames_p[f2]
            for o in range(nO):
                op = obs_p[o]
                ov = obs_v[o]
                oa = obs_a[o]

                # Volumetric segment-to-point capsule distance
                d_surf, c1, c2, p_r_surf, p_h_surf, s_star, t_star = segment_to_segment_distance(
                    p1, p2, op, op, r_robot, r_human, atol
                )

                # Linearly interpolate translational Jacobian and link velocity along cylinder centerline
                Jlin_star = (1.0 - s_star) * Jlins[f1] + s_star * Jlins[f2]
                dJlin_star = (1.0 - s_star) * dJlins[f1] + s_star * dJlins[f2]
                vlin_star = (1.0 - s_star) * frames_vlin[f1] + s_star * frames_vlin[f2]

                h, row_vec, bound, d, vr, vh = compute_h_and_constraints_numba(
                    p_r_surf, p_h_surf, vlin_star, ov, Tr, a_s, C, oa, atol,
                    Jlin_star, dJlin_star, dq, gamma, HAS_CBF, delta_H, tc,
                    sigma_psi, beta
                )

                if keypoint_to_log >= 0:
                    if o == min(keypoint_to_log, nO - 1) and s == nSeg - 1:
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
                    for j in range(A.shape[1]):
                        A[row, j] = 0.0
                    for j in range(nq):
                        A[row, j] = row_vec[j]
                    c[row] = bound
                    row += 1
    else:
        nF = frames_p.shape[0]
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
                    p_bt, op, vlin, ov, Tr, a_s, C, oa, atol, Jlin, dJlin, dq, gamma, HAS_CBF, delta_H, tc,
                    sigma_psi, beta
                )

                # Logging keypoint selection
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
                    for j in range(A.shape[1]):
                        A[row, j] = 0.0
                    for j in range(nq):
                        A[row, j] = row_vec[j]
                    c[row] = bound
                    row += 1

    return row, hmin, dmin, vr_min, vh_min, htest, dtest, i_h, i_d


# Convenient alias
fill_cbf_rows = append_cbf_rows_loop



# ------------------------------------------------------------
# 3) Objective Assembly: Assemble Hessian P and Linear Vector b
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def assemble_objective_parts_inplace(
    P2, b_pos, b_vel, b_scaling,
    q, dq,
    nominal_q, nominal_Dq,
    Dtraj, Tc, ref_scaling,
):
    """
    Assemble tracking cost components for position, velocity, and trajectory scaling.

    Cost terms:
        Position:  0.5 * ||q(k) + Tc*dq + 0.5*Tc^2*ddq - nominal_q||^2
        Velocity:  0.5 * ||dq(k) + Tc*ddq - Dtraj*nominal_Dq - Tc*DDtraj*nominal_Dq||^2
        Scaling:   0.5 * (Dtraj + Tc*DDtraj - ref_scaling)^2
    """
    nq = q.size
    n_vars = P2.shape[0]
    for i in range(n_vars):
        for j in range(n_vars):
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
# 4) High-Level: In-Place QP Matrix Assembly
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def assemble_qp_inplace(
    P2, b_pos, b_vel, b_scaling,
    A, c,
    FreePos, ForcedPos, FreeVel, ForcedVel,
    q, dq,
    nominal_q, nominal_Dq,
    Dtraj, Tc,
    q_min, q_max,
    Dq_max, DDq_max, delta_q_max,
    frames_p, frames_vlin, Jlins, dJlins, obs_p, obs_v, obs_a,
    Tr, a_s, C, gamma, DDtraj_max, atol, ref_scaling, HAS_CBF, keypoint_to_log,
    delta_H=0.0,
    sigma_psi=np.zeros(4, dtype=np.float64),
    beta=3.291,
    capsule_segments=np.empty((0, 2), dtype=np.int64),
    r_robot=0.0,
    r_human=0.0,
):
    """
    Zero out QP matrices and populate all constraints and objective components in-place.
    """
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

    row = fill_pos_rows(A, c, row, nq, FreePos, ForcedPos, x0, q_min, q_max)
    row = fill_tube_rows(A, c, row, nq, FreePos, ForcedPos, x0, nominal_q, delta_q_max)
    row = fill_vel_rows(A, c, row, nq, FreeVel, ForcedVel, x0, Dq_max)
    row = fill_acc_rows(A, c, row, nq, DDq_max)

    if frames_p.size != 0 and obs_p.size != 0:
        row, hmin, dmin, vr_min, vh_min, htest, dtest, i_h, i_d = append_cbf_rows_loop(
            A, c, row, frames_p, frames_vlin, obs_p, obs_v, obs_a, Jlins, dJlins, dq, Tr, a_s, C, gamma, atol, HAS_CBF,
            keypoint_to_log, delta_H, tc=Tc,
            sigma_psi=sigma_psi, beta=beta,
            capsule_segments=capsule_segments, r_robot=r_robot, r_human=r_human,
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
        i_h = 0
        i_d = 0

    assemble_objective_parts_inplace(P2, b_pos, b_vel, b_scaling, q, dq, nominal_q, nominal_Dq, Dtraj, Tc, ref_scaling)
    return row, hmin, dmin, vr_min, vh_min, htest, dtest, i_h, i_d


# Convenient alias
fill_cbf_rows = append_cbf_rows_loop


