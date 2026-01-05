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
        self.n_constraints = 2 * 2 * nq + 18 * len(self.frames_ids)
        print(f"n_constraints: {self.n_constraints}")
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

    # def cbf_ensemble(
    #     self,
    #     translation_bt: np.ndarray,
    #     vel_lineare: np.ndarray,
    #     Jlin: np.ndarray,
    #     dJlin: np.ndarray,
    #     obstacle_positions: np.ndarray,
    #     t: float,
    # ):
    #     """
    #     Assemble CBF-based inequality constraints:
    #
    #         A ddq <= c
    #
    #     where each row encodes one CBF constraint for one obstacle.
    #
    #     This also updates obstacle_positions in-place according to the
    #     time-varying motion pattern, and returns h_min (minimum h).
    #     """
    #     nq = self.model.nq
    #     A = np.empty((0, nq))
    #     c = np.empty((0, 1))
    #
    #     h_min = np.inf
    #
    #     for i, obs_pos in enumerate(obstacle_positions):
    #         # Simple circular-ish motion for the obstacle as in the original code
    #         w1 = 2 * np.pi / 2.0
    #         w2 = 2 * np.pi / 2.1
    #         obs_pos[0] = 0.8 - 0.25 * np.sin(w1 * t)
    #         obs_pos[1] = 0.4 + 0.1 * np.sin(w2 * t)
    #
    #         v_obs = np.array([0.0, 0.0, 0.0])
    #         v_obs[0] = -0.25 * np.cos(w1 * t) * w1
    #         v_obs[1] = 0.1 * np.cos(w2 * t) * w2
    #
    #         r = translation_bt - obs_pos
    #         distance = np.linalg.norm(r)
    #         u_hr = r / distance
    #
    #         v_h_scalar = float(u_hr @ v_obs)
    #         v_rel = float(u_hr @ vel_lineare)
    #
    #         # CBF scalar
    #         h = compute_h(d=distance, v=v_rel, v_h=v_h_scalar)
    #         if h < h_min:
    #             h_min = h
    #
    #         # Range dynamics and Jacobians
    #         f, g = range_state_derivative(vel_lineare, v_obs)
    #         dh_dd, dh_dv, dh_dvh = jacobian_h(distance, v_rel, v_h_scalar)
    #         Jh_psi = np.array([dh_dd, dh_dv, dh_dvh]).reshape(1, -1)
    #
    #         Jpsi_chi = jacobian_psi(translation_bt, obs_pos, vel_lineare, v_obs)
    #
    #         Lie_f_h = Jh_psi @ Jpsi_chi @ f          # shape (1,)
    #         Lie_g_h = Jh_psi @ Jpsi_chi @ g          # shape (1, 3)
    #
    #         # Map to joint space via Jlin
    #         A_i = (Lie_g_h @ Jlin).reshape(1, -1)
    #         c_i = (-Lie_g_h @ dJlin @ self.dq - Lie_f_h - self.gamma * h).reshape(1, -1)
    #
    #         A = np.append(A, A_i, axis=0)
    #         c = np.append(c, c_i, axis=0)
    #
    #     return A, c.flatten(), float(h_min), obstacle_positions

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
        cbf_enabled: bool = True,
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


        if cbf_enabled and len(obstacle_positions) > 0:
            row, h_min, d_min, vr_min, vh_min = assemble_qp_PID_problem(
                # outputs (in-place)
                self.A, self.c,
                # inputs
                self.FreeVel, self.ForcedVel,
                self.q, self.dq,
                Dq_max, DDq_max,
                # CBF inputs (optional; pass empty arrays if unused)
                frames_p, frames_v, Jlins, dJlins, obstacle_positions, obstacle_velocities, obstacle_accelerations,
                Tr, a_s, C, gamma, 1e-12
            )

        # ---------------------------- QP solve ------------------------------- #
        if cbf_enabled and self.A.shape[0] > 0:
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
