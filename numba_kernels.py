# numba_kernels.py
# Pattern A: NUMBA for assembly-only (no solver calls here)

import numpy as np
from numba import njit

# ------------------------------------------------------------
# Primitive blocks
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def build_free_forced_one_step(Ts: float, nq: int):
    I = np.eye(nq)
    ForcedPos = 0.5 * (Ts ** 2) * I
    FreePos   = np.hstack((I, Ts * I))
    ForcedVel = Ts * I
    FreeVel   = np.hstack((np.zeros_like(I), I))
    return FreePos, ForcedPos, FreeVel, FreeVel  # NOTE: FreeVel returned twice for ABI stability (no second alloc)


# ------------------------------------------------------------
# Constraint rows (assembled in-place)
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def fill_scaling_rows(A, c, row, nq, Tc, Dtraj, DDtraj_max):
    # [-0..0, -Tc]*[ddq,DDtraj] <= -(1 - Dtraj)
    for j in range(nq):
        A[row, j] = 0.0
    A[row, nq] = -Tc
    c[row] = -(1.0 - Dtraj)
    row += 1

    # [0..0, +Tc]*[...] <= -Dtraj
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
    # q(k+1) tube around nominal
    Fx = FreePos @ x0  # shape (nq,)
    # lower: -ForcedPos*ddq <= -nominal - delta + Fx
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = -ForcedPos[i, j]
        c[row + i] = -nominal_q[i] - delta_q_max[i] + Fx[i]
    row += nq

    # upper: +ForcedPos*ddq <= +nominal - delta - Fx
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = +ForcedPos[i, j]
        c[row + i] = +nominal_q[i] - delta_q_max[i] - Fx[i]
    row += nq
    return row


@njit(cache=True, fastmath=True)
def fill_vel_rows(A, c, row, nq, FreeVel, ForcedVel, x0, Dq_max):
    Fx = FreeVel @ x0  # (nq,)

    # -ForcedVel*ddq <= -Dq_max + Fx
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = -ForcedVel[i, j]
        c[row + i] = -Dq_max[i] + Fx[i]
    row += nq

    # +ForcedVel*ddq <= -Dq_max - Fx
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = +ForcedVel[i, j]
        c[row + i] = -Dq_max[i] - Fx[i]
    row += nq
    return row


@njit(cache=True, fastmath=True)
def fill_acc_rows(A, c, row, nq, DDq_max):
    # -I*ddq <= -DDq_max
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = -1.0 if i == j else 0.0
        c[row + i] = -DDq_max[i]
    row += nq

    # +I*ddq <= -DDq_max
    for i in range(nq):
        for j in range(nq):
            A[row + i, j] = +1.0 if i == j else 0.0
        c[row + i] = -DDq_max[i]
    row += nq
    return row


# ------------------------------------------------------------
# Optional: CBF rows, using your own jit-safe primitive
# ------------------------------------------------------------
# Expected signature (you already have this):
# @njit
# def compute_h_and_constraints_numba(p_bt, op, v_lin, ov, Tr, a_s, C, oa, atol, Jlin, dJlin, dq, gamma):
#     return h, row_vec(=shape (nq,)), bound
try:
    from ssm_cbf_acc import compute_h_and_constraints_numba  # ensure this is @njit
    HAS_CBF = True
except Exception:
    HAS_CBF = False

@njit(cache=True, fastmath=True)
def append_cbf_rows_loop(
    A, c, row,
    frames_p, frames_vlin,  # (nF,3), (nF,3)
    obs_p, obs_v, obs_a,    # (nO,3)
    Jlins, dJlins, dq,      # (nF,3,nq)
    Tr, a_s, C, gamma, atol
):
    hmin = 1e9
    dmin = 1e9
    vrel_min = 0.0  # unused placeholder
    if not HAS_CBF:
        return row, hmin

    nF = frames_p.shape[0]
    nO = obs_p.shape[0]
    nq = dq.size

    for f in range(nF):
        p_bt = frames_p[f]
        vlin = frames_vlin[f]
        Jlin = Jlins[f]
        dJlin = dJlins[f]
        for o in range(nO):
            op = obs_p[o]; ov = obs_v[o]; oa = obs_a[o]
            h, row_vec, bound, d, vrel = compute_h_and_constraints_numba(
                p_bt, op, vlin, ov, Tr, a_s, C, oa, atol, Jlin, dJlin, dq, gamma
            )
            if h < hmin:
                hmin = h
            if d < dmin:
                dmin = d
            if vrel < vrel_min:
                vrel_min = vrel
            for j in range(nq):
                A[row, j] = row_vec[j]
            c[row] = bound
            row += 1
    return row, hmin, dmin, vrel_min


# ------------------------------------------------------------
# Objective assembly (only the P2, b1, b2, b3 parts)
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def assemble_objective_parts_inplace(P2, b1, b2, b3,
                                     q, dq,
                                     nominal_q, nominal_Dq,
                                     Dtraj, Tc):
    nq = q.size

    # zero
    for i in range(nq + 1):
        for j in range(nq + 1):
            P2[i, j] = 0.0
        b1[i] = 0.0
        b2[i] = 0.0
        b3[i] = 0.0

    # P2
    ndq_dot = 0.0
    for i in range(nq):
        val = -(Tc * Tc) * nominal_Dq[i]
        P2[i, nq] = val
        P2[nq, i] = val
        P2[i,i] = Tc**2
        ndq_dot += nominal_Dq[i] * nominal_Dq[i]
    P2[nq, nq] = (Tc * Tc) * ndq_dot

    # b1 (tracking position)
    half_T2 = 0.5 * Tc * Tc
    for i in range(nq):
        b1[i] = (nominal_q[i] - q[i] - dq[i] * Tc) * half_T2
    # b1[-1] remains 0

    # b2 (tracking velocity scaled by Dtraj)
    tmp_dot = 0.0
    for i in range(nq):
        val = (nominal_Dq[i] * Dtraj - dq[i]) * Tc
        b2[i] = val
        tmp_dot += (nominal_Dq[i] * Dtraj - dq[i]) * (nominal_Dq[i] * Tc)
    b2[nq] = -tmp_dot

    # b3 (penalize deviation of Dtraj from 1)
    b3[nq] = -Tc * (Dtraj - 1.0)


# ------------------------------------------------------------
# High-level: assemble ALL constraints, and objective parts
# ------------------------------------------------------------
@njit(cache=True, fastmath=True)
def assemble_qp_inplace(
    # outputs (in-place)
    P2, b1, b2, b3,
    A, c,
    # inputs
    FreePos, ForcedPos, FreeVel, ForcedVel,
    q, dq,
    nominal_q, nominal_Dq,
    Dtraj, Tc,
    Dq_max, DDq_max, delta_q_max,
    # CBF inputs (optional; pass empty arrays if unused)
    frames_p, frames_vlin, Jlins, dJlins, obs_p, obs_v, obs_a,
    Tr, a_s, C, gamma, DDtraj_max, atol
):
    nq = q.size
    # zero A, c
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = 0.0
        c[i] = 0.0

    # constraint rows
    row = 0
    row = fill_scaling_rows(A, c, row, nq, Tc, Dtraj, DDtraj_max=DDtraj_max)  # placeholder overwritten below
    # overwrite last line's rhs to -DDtraj_max using c[row_idx] once caller fixes it if needed

    x0 = np.empty(nq * 2)
    for i in range(nq):
        x0[i] = q[i]
        x0[nq + i] = dq[i]

    row = fill_tube_rows(A, c, row, nq, FreePos, ForcedPos, x0, nominal_q, delta_q_max)
    row = fill_vel_rows(A, c, row, nq, FreeVel, ForcedVel, x0, Dq_max)
    row = fill_acc_rows(A, c, row, nq, DDq_max)

    # CBF rows (if any)
    if frames_p.size != 0 and obs_p.size != 0 and HAS_CBF:
        row, hmin, dmin, vrel_min = append_cbf_rows_loop(
            A, c, row, frames_p, frames_vlin, obs_p, obs_v, obs_a, Jlins, dJlins, dq, Tr, a_s, C, gamma, atol
        )
    else:
        hmin = 1e9
        dmin = 1e9
        vrel_min = 1e9  # unused placeholder
    # objective parts
    assemble_objective_parts_inplace(P2, b1, b2, b3, q, dq, nominal_q, nominal_Dq, Dtraj, Tc)

    return row, hmin, dmin, vrel_min