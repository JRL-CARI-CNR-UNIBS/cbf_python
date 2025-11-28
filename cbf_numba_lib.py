# cbf_numba_lib.py
#
# Numba-accelerated CBF / kinematic helper functions used by the UR10 main loop.

import numpy as np
from numba import njit
import pinocchio as pin
from numba_kernels import fill_vel_rows, fill_acc_rows, append_cbf_rows_loop

# Default constants (kept here so JITted functions have literal defaults)
C_DEFAULT = 0.25   # [m]
Tr_DEFAULT = 0.15  # [s]
A_S_DEFAULT = 2.5  # [m/s^2]


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
# @njit(cache=True, fastmath=True)
def compute_q_ref_from_goal(goal_pose: pin.SE3, model, data, q, tool_frame_id) -> np.ndarray:
    """
    Compute a reference joint configuration whose tool frame matches goal_pose
    as closely as possible using a simple IK loop in Pinocchio.

    Returns
    -------
    q_ref : (nq,) array
        Reference joint configuration.
    """
    model = model
    data = data

    # Start from current configuration (you can change the initial guess)
    q = q.copy()

    eps = 1e-4
    IT_MAX = 100
    alpha = 0.5  # step size for IK updates

    for _ in range(IT_MAX):
        # Forward kinematics for this q
        pin.framesForwardKinematics(model, data, q)
        Tbt = data.oMf[tool_frame_id]

        # SE(3) error between current and desired tool pose
        # (goal in base frame, same convention you use in the controller)
        T_err = Tbt.inverse() * goal_pose
        err6 = pin.log(T_err)  # 6D error (translation + rotation)

        if np.linalg.norm(err6) < eps:
            break

        # 6D Jacobian of the tool frame
        J6 = pin.computeFrameJacobian(
            model,
            data,
            q,
            tool_frame_id,
            pin.ReferenceFrame.LOCAL
        )

        # Simple damped least-squares step in joint space
        dq = -alpha * damped_pinv_svd(J6) @ err6
        q = pin.integrate(model, q, dq)

    return q
