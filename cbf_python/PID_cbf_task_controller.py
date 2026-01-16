import pinocchio as pin
import numpy as np

from cbf_numba_lib import (
    assemble_qp_PID_problem, compute_q_ref_from_goal
)
import quadprog

from numba_kernels import build_free_forced_one_step

Tr = 0.5
a_s = 4.5
C = 0.25
gamma = 5.0
Tc = 2e-3
Dq_max: np.ndarray = np.pi * np.array([1, 1, 1, 1, 1, 1], dtype=np.float64) * np.pi
DDq_max: np.ndarray = np.pi * np.array([1, 1, 1, 1, 1, 1], dtype=np.float64) * np.pi * 5.0



class UR10CBFController:
    """
    Simple Cartesian-space CBF controller for UR10.

    - Holds the robot state (q, dq, ddq).
    - At each .step() call, computes desired Cartesian acceleration,
      assembles the QP (cost + CBF constraints), solves for ddq,
      and integrates q, dq.
    """

    def __init__(
        self,
        model: pin.Model,
        tool_frame_name: str,
        frames_ids,
        Tc: float,
        Kp_tra: np.ndarray,
        Kd_tra: np.ndarray,
        Kp_rot: np.ndarray,
        Kd_rot: np.ndarray,
        gamma: float = 5,
        useCbf = True,
    ):
        self.model = model
        self.data = model.createData()
        self.tool_frame_id = model.getFrameId(tool_frame_name)
        self.Tc = Tc

        # gains
        self.Kp_tra = Kp_tra
        self.Kd_tra = Kd_tra
        self.Kp_rot = Kp_rot
        self.Kd_rot = Kd_rot

        self.gamma = gamma
        # NUMBA-prebuilt blocks
        self.FreePos, self.ForcedPos, self.FreeVel, self.ForcedVel = \
            build_free_forced_one_step(Tc, self.model.nq)
        # state
        self.q = np.zeros(self.model.nq)
        self.dq = np.zeros(self.model.nq)
        self.ddq = np.zeros(self.model.nq)
        self.frames_ids = frames_ids
        # constraints (upper bound)
        nq = self.model.nq
        # print(f"n_constraints: {self.n_constraints}")
        self.useCbf = useCbf
        if useCbf:
            self.n_constraints = 2 * 2 * nq + 18 * len(self.frames_ids)
        else:
            self.n_constraints = 2 * 2 * nq
        self.A = np.zeros((self.n_constraints, nq), dtype=np.float64)
        self.c = np.zeros(self.n_constraints, dtype=np.float64)

    # ---------------------------------------------------------------------- #
    #                         State management                               #
    # ---------------------------------------------------------------------- #

    def reset_state(self, q0: np.ndarray, dq0: np.ndarray | None = None):
        """Reset joint state."""
        self.q = q0.copy()
        if dq0 is None:
            self.dq = np.zeros_like(q0)
        else:
            self.dq = dq0.copy()
        self.ddq = np.zeros_like(q0)

    # ---------------------------------------------------------------------- #
    #                    QP matrix assembly helpers                          #
    # ---------------------------------------------------------------------- #

    def matrix_ensemble(
        self,
        J: np.ndarray,
        dJ: np.ndarray,
        dq: np.ndarray,
        dtwist_tool: np.ndarray,
    ):
        """
        Assemble non-CBF part of the QP:
          minimize 0.5 * ddq^T P ddq + b^T ddq

        Returns
        -------
        P : (nq, nq)
        b : (nq,)
        """
        P = J.T @ J
        b = (J.T @ (dtwist_tool - dJ @ dq)).flatten()
        return P, b

    # ---------------------------------------------------------------------- #
    #                             Control step                               #
    # ---------------------------------------------------------------------- #

    def step(
        self,
        t: float,
        goal_pose: pin.SE3,
        twist_goal: np.ndarray,
        goal_dtwist: np.ndarray,
        obstacle_positions,
        obstacle_velocities,
        obstacle_accelerations,
    ):
        """
        Perform one control step:

        Parameters
        ----------
        t : float
            Current simulation time [s].
        goal_pose : SE3
            Desired tool pose at this time.
        twist_goal : (6,) array
            Desired tool twist (linear + angular).
        goal_dtwist : (6,) array
            Desired tool twist acceleration (currently unused, but kept for API).
        obstacle_positions : List[3-array]
            Obstacle positions (will be updated in-place by cbf_ensemble).
        cbf_enabled : bool
            Whether to enforce CBF constraints.

        Returns
        -------
        out : dict
            {
                "q":             current joint positions after integration,
                "dq":            current joint velocities,
                "ddq":           joint accelerations just computed,
                "Tbt":           current tool SE3,
                "h_min":         minimum CBF value over obstacles,
                "obs_pos":       updated obstacle positions,
            }
        """
        model = self.model
        data = self.data

        # --- Forward kinematics ---
        pin.framesForwardKinematics(model, data, self.q)
        pin.computeForwardKinematicsDerivatives(model, data, self.q, self.dq, self.ddq)

        Tbt = data.oMf[self.tool_frame_id]
        translation_bt = Tbt.translation
        Rbt = Tbt.rotation.copy()

        Rbg = goal_pose.rotation.copy()
        G = goal_pose.translation

        # Orientation error
        Rtg = Rbt.T @ Rbg
        error_rot = Rbt @ pin.log3(Rtg)

        # Current twist
        twist = pin.getFrameVelocity(
            model, data, self.tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        vel_lin = twist.linear
        vel_ang = twist.angular

        # Jacobians
        J = pin.computeFrameJacobian(
            model, data, self.q, self.tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        dJ = pin.frameJacobianTimeVariation(
            model, data, self.q, self.dq, self.tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        Jlin = J[:3, :]
        dJlin = dJ[:3, :]

        # Desired Cartesian accelerations
        acc_lin = self.Kp_tra * (G - translation_bt) + self.Kd_tra * (twist_goal[:3] - vel_lin)
        acc_ang = self.Kp_rot * error_rot + self.Kd_rot * (twist_goal[3:] - vel_ang)
        dtwist_tool = np.hstack([acc_lin, acc_ang])  # + goal_dtwist if desired

        # ---------------------------- QP assembly ---------------------------- #
        P, b = self.matrix_ensemble(J, dJ, self.dq, dtwist_tool)


        h_min = np.inf
        d_min = np.inf
        nq = model.nq
        vr_min = np.inf
        vh_min = np.inf
        nF = len(self.frames_ids)
        frames_p = np.zeros((nF, 3), dtype=np.float64)
        frames_v = np.zeros((nF, 3), dtype=np.float64)
        Jlins = np.zeros((nF, 3, nq), dtype=np.float64)
        dJlins = np.zeros((nF, 3, nq), dtype=np.float64)

        for i, f_id in enumerate(self.frames_ids):
            Tbt = self.data.oMf[f_id]
            frames_p[i, :] = Tbt.translation
            twist = pin.getFrameVelocity(self.model, self.data, f_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            frames_v[i, :] = twist.linear
            J = pin.computeFrameJacobian(self.model, self.data, self.q, f_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            dJ = pin.frameJacobianTimeVariation(self.model, self.data, self.q, self.dq, f_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jlins[i, :, :]  = J[:3, :]
            dJlins[i, :, :] = dJ[:3, :]


        if self.useCbf and len(obstacle_positions) > 0:
            row, h_min, d_min, vr_min, vh_min = assemble_qp_PID_problem(
                # outputs (in-place)
                self.A, self.c,
                # inputs
                self.FreeVel, self.ForcedVel,
                self.q, self.dq,
                Dq_max, DDq_max,
                # CBF inputs (optional; pass empty arrays if unused)
                frames_p, frames_v, Jlins, dJlins, obstacle_positions, obstacle_velocities, obstacle_accelerations,
                Tr, a_s, C, gamma, 1e-12, self.useCbf
            )

        # ---------------------------- QP solve ------------------------------- #
        if self.useCbf and self.A.shape[0] > 0:
            try:
                ddq, *_ = quadprog.solve_qp(
                    P,
                    b,
                    self.A.T,
                    self.c,
                    0,  # no equality constraints
                )
            except ValueError as err:
                if "constraints are inconsistent" in str(err):
                    print("[QP] infeasible – applying fallback damping.")
                    ddq = -10.0 * self.dq
                else:
                    raise
        else:
            # No CBF, or no constraints: damped least-squares solution
            ddq = damped_pinv_svd(J) @ (dtwist_tool - dJ @ self.dq)

        # ---------------------------- Integrate ------------------------------ #
        self.q = self.q + self.dq * self.Tc + 0.5 * ddq * self.Tc ** 2
        self.dq = self.dq + ddq * self.Tc
        self.ddq = ddq

        # Recompute pose for visualization
        pin.framesForwardKinematics(model, data, self.q)
        Tbt_new = data.oMf[self.tool_frame_id]
        frames_p[-1, :] = Tbt_new.translation
        twist = pin.getFrameVelocity(self.model, self.data, self.tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        frames_v[-1, :] = twist.linear
        # -------------------- Joint-space trajectory error ------------------- #

        traj_err = float(np.linalg.norm(goal_pose.translation.tolist() - frames_p[-1, :]))  # NEW: ||q - q_ref||

        return {
            "q": self.q.copy(),
            "dq": self.dq.copy(),
            "ddq": self.ddq.copy(),
            "Tbt": Tbt_new,
            "h_min": float(h_min),
            "d_min": float(d_min),
            "vr_min": float(vr_min),
            "vh_min": float(vh_min),
            "obs_pos": obstacle_positions,
            "trajectory_error": traj_err,
            "end_effector_pos": frames_p[-1, :].copy(),
            "end_effector_vel": frames_v[-1, :].copy(),
        }
