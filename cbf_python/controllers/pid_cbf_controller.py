"""
Cartesian PID task controller with Control Barrier Function (CBF) constraints.
"""

from __future__ import annotations
from typing import Sequence, List, Dict, Any, Optional, Union
import numpy as np
import quadprog
import pinocchio as pin

from cbf_python.controllers.kernels.numba_kernels import build_free_forced_one_step
from cbf_python.controllers.kernels.cbf_numba_lib import assemble_qp_PID_problem, damped_pinv_svd


class UR10CBFController:
    """Cartesian-space CBF controller for UR10.

    - Holds robot state (q, dq, ddq).
    - In each .step() call, computes desired Cartesian acceleration via Cartesian PID,
      assembles QP with CBF safety constraints, solves for ddq, and integrates.
    """

    def __init__(
        self,
        model: pin.Model,
        tool_frame_name: str = "ur10e_wrist_3_joint",
        frames_ids: Optional[Sequence[int]] = None,
        Tc: float = 2e-3,
        Kp_tra: Union[float, np.ndarray] = 40.0,
        Kd_tra: Union[float, np.ndarray] = 12.0,
        Kp_rot: Union[float, np.ndarray] = 20.0,
        Kd_rot: Union[float, np.ndarray] = 8.0,
        gamma: float = 5.0,
        useCbf: bool = True,
        Tr: float = 0.5,
        a_s: float = 4.5,
        C: float = 0.25,
        Dq_max: Optional[np.ndarray] = None,
        DDq_max: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.data = model.createData()
        self.tool_frame_id = model.getFrameId(tool_frame_name)
        self.frames_ids = list(frames_ids) if frames_ids is not None else [self.tool_frame_id]
        self.Tc = float(Tc)

        # Gains
        self.Kp_tra = np.asarray(Kp_tra, dtype=float)
        self.Kd_tra = np.asarray(Kd_tra, dtype=float)
        self.Kp_rot = np.asarray(Kp_rot, dtype=float)
        self.Kd_rot = np.asarray(Kd_rot, dtype=float)

        self.gamma = float(gamma)
        self.Tr = float(Tr)
        self.a_s = float(a_s)
        self.C = float(C)

        nq = self.model.nq
        self.Dq_max = Dq_max if Dq_max is not None else np.pi * np.ones(nq, dtype=np.float64) * np.pi
        self.DDq_max = DDq_max if DDq_max is not None else self.Dq_max * 5.0

        self.FreePos, self.ForcedPos, self.FreeVel, self.ForcedVel = build_free_forced_one_step(
            self.Tc, nq
        )

        self.q = np.zeros(nq, dtype=np.float64)
        self.dq = np.zeros(nq, dtype=np.float64)
        self.ddq = np.zeros(nq, dtype=np.float64)

        self.useCbf = useCbf
        if useCbf:
            self.n_constraints = 2 * 2 * nq + 18 * len(self.frames_ids) + 4
        else:
            self.n_constraints = 2 * 2 * nq

        self.A = np.zeros((self.n_constraints, nq), dtype=np.float64)
        self.c = np.zeros(self.n_constraints, dtype=np.float64)

    def reset_state(self, q0: np.ndarray, dq0: Optional[np.ndarray] = None) -> None:
        """Reset internal joint state."""
        self.q = np.array(q0, dtype=np.float64).copy()
        if dq0 is None:
            self.dq = np.zeros_like(self.q)
        else:
            self.dq = np.array(dq0, dtype=np.float64).copy()
        self.ddq = np.zeros_like(self.q)

    def matrix_ensemble(
        self,
        J: np.ndarray,
        dJ: np.ndarray,
        dq: np.ndarray,
        dtwist_tool: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble nominal tracking quadratic cost."""
        P = J.T @ J
        b = (J.T @ (dtwist_tool - dJ @ dq)).flatten()
        return P, b

    def step(
        self,
        goal_pose: pin.SE3,
        twist_goal: np.ndarray,
        obstacle_positions: np.ndarray,
        obstacle_velocities: np.ndarray,
        obstacle_accelerations: np.ndarray,
        t: float = 0.0,
        goal_dtwist: Optional[np.ndarray] = None,
        gamma: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute one Cartesian PID CBF control step."""
        if goal_dtwist is None:
            goal_dtwist = np.zeros(6, dtype=np.float64)
        if gamma is not None:
            self.gamma = float(gamma)

        model = self.model
        data = self.data

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

        twist = pin.getFrameVelocity(
            model, data, self.tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        vel_lin = twist.linear
        vel_ang = twist.angular

        J = pin.computeFrameJacobian(
            model, data, self.q, self.tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        dJ = pin.frameJacobianTimeVariation(
            model, data, self.q, self.dq, self.tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        acc_lin = self.Kp_tra * (G - translation_bt) + self.Kd_tra * (twist_goal[:3] - vel_lin)
        acc_ang = self.Kp_rot * error_rot + self.Kd_rot * (twist_goal[3:] - vel_ang)
        dtwist_tool = np.hstack([acc_lin, acc_ang])

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
            Tf = self.data.oMf[f_id]
            frames_p[i, :] = Tf.translation
            tw = pin.getFrameVelocity(self.model, self.data, f_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            frames_v[i, :] = tw.linear
            Jf = pin.computeFrameJacobian(self.model, self.data, self.q, f_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            dJf = pin.frameJacobianTimeVariation(self.model, self.data, self.q, self.dq, f_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jlins[i, :, :] = Jf[:3, :]
            dJlins[i, :, :] = dJf[:3, :]

        if self.useCbf and len(obstacle_positions) > 0:
            row, h_min, d_min, vr_min, vh_min = assemble_qp_PID_problem(
                self.A, self.c,
                self.FreeVel, self.ForcedVel,
                self.q, self.dq,
                self.Dq_max, self.DDq_max,
                frames_p, frames_v, Jlins, dJlins,
                obstacle_positions, obstacle_velocities, obstacle_accelerations,
                self.Tr, self.a_s, self.C, self.gamma, 1e-12, self.useCbf,
            )

        if self.useCbf and self.A.shape[0] > 0:
            try:
                ddq, *_ = quadprog.solve_qp(P, b, self.A.T, self.c, 0)
            except ValueError as err:
                if "constraints are inconsistent" in str(err):
                    # Infeasible fallback: damping
                    ddq = -10.0 * self.dq
                else:
                    raise
        else:
            ddq = damped_pinv_svd(J) @ (dtwist_tool - dJ @ self.dq)

        self.q = self.q + self.dq * self.Tc + 0.5 * ddq * (self.Tc ** 2)
        self.dq = self.dq + ddq * self.Tc
        self.ddq = ddq

        pin.framesForwardKinematics(model, data, self.q)
        Tbt_new = data.oMf[self.tool_frame_id]
        frames_p[-1, :] = Tbt_new.translation
        tw_end = pin.getFrameVelocity(self.model, self.data, self.tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        frames_v[-1, :] = tw_end.linear

        traj_err = float(np.linalg.norm(goal_pose.translation - frames_p[-1, :]))

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

    def close(self) -> None:
        """Release allocated arrays and resources."""
        self.model = None
        self.data = None
        self.FreePos = self.ForcedPos = None
        self.FreeVel = self.ForcedVel = None
        self.A = None
        self.c = None
        self.q = self.dq = self.ddq = None
