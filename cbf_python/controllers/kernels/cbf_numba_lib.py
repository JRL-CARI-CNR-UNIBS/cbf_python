"""
Numba-accelerated helper routines for Cartesian PID and CBF QP problem assembly.
"""

from __future__ import annotations
import numpy as np
from numba import njit
import pinocchio as pin

from cbf_python.controllers.kernels.numba_kernels import (
    fill_pos_rows,
    fill_vel_rows,
    fill_acc_rows,
    append_cbf_rows_loop,
)

# Default constant values
C_DEFAULT: float = 0.25      # Safety margin [m]
TR_DEFAULT: float = 0.15     # Reaction time [s]
A_S_DEFAULT: float = 2.5     # Maximum human deceleration [m/s^2]


def damped_pinv_svd(J: np.ndarray, damping: float = 1e-4) -> np.ndarray:
    """Compute damped pseudoinverse via Singular Value Decomposition."""
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_damped = S / (S ** 2 + damping ** 2)
    return Vt.T @ np.diag(S_damped) @ U.T


@njit(cache=True, fastmath=True)
def assemble_qp_PID_problem(
    # Outputs (in-place)
    A, c,
    # Inputs
    FreePos, ForcedPos,
    FreeVel, ForcedVel,
    q, dq,
    q_min, q_max,
    Dq_max, DDq_max,
    # CBF inputs
    frames_p, frames_vlin, Jlins, dJlins, obs_p, obs_v, obs_a,
    Tr, a_s, C, gamma, atol, use_CBF,
    delta_H=0.0,
):
    """Assemble constraints for Cartesian PID CBF controller."""
    nq = q.size
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = 0.0
        c[i] = 0.0

    row = 0
    x0 = np.empty(nq * 2)
    for i in range(nq):
        x0[i] = q[i]
        x0[nq + i] = dq[i]

    row = fill_pos_rows(A, c, row, nq, FreePos, ForcedPos, x0, q_min, q_max)
    row = fill_vel_rows(A, c, row, nq, FreeVel, ForcedVel, x0, Dq_max)
    row = fill_acc_rows(A, c, row, nq, DDq_max)

    if frames_p.size != 0 and obs_p.size != 0:
        row, hmin, dmin, vr_min, vh_min, htest, dtest, i_h, i_d = append_cbf_rows_loop(
            A, c, row, frames_p, frames_vlin, obs_p, obs_v, obs_a, Jlins, dJlins, dq, Tr, a_s, C, gamma, atol, use_CBF, -1,
            delta_H
        )
    else:
        hmin = 1e9
        dmin = 1e9
        vr_min = 1e9
        vh_min = 1e9

    return row, hmin, dmin, vr_min, vh_min


def compute_q_ref_from_goal(
    goal_pose: pin.SE3,
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    tool_frame_id: int,
) -> np.ndarray:
    """Compute reference joint configuration matching goal_pose via Pinocchio numerical inverse kinematics."""
    q_out = q.copy()
    eps = 1e-4
    IT_MAX = 100
    alpha = 0.5

    for _ in range(IT_MAX):
        pin.framesForwardKinematics(model, data, q_out)
        Tbt = data.oMf[tool_frame_id]

        T_err = Tbt.inverse() * goal_pose
        err6 = pin.log(T_err)

        if np.linalg.norm(err6) < eps:
            break

        J6 = pin.computeFrameJacobian(
            model, data, q_out, tool_frame_id, pin.ReferenceFrame.LOCAL
        )
        dq = -alpha * damped_pinv_svd(J6) @ err6
        q_out = pin.integrate(model, q_out, dq)

    return q_out
