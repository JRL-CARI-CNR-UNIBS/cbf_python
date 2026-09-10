"""
Base Control Barrier Function (CBF) optimal task controller and configuration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np
import quadprog
import pinocchio as pin

from cbf_python.controllers.kernels.numba_kernels import (
    build_free_forced_one_step,
    assemble_qp_inplace,
)
from cbf_python.controllers.velocity_scaling import compute_velocity_scaling_for_human_proximity


@dataclass
class ControllerConfig:
    """Configuration parameters for the CBF optimal controller."""

    Tc: float = 2e-3                          # Control period [s]
    C: float = 0.25                           # Safety margin [m]
    Tr: float = 0.15                          # Reaction time [s]
    a_s: float = 2.5                          # Robot maximum deceleration [m/s^2]
    gamma: float = 5.0                        # CBF class-K gain
    max_obstacles: int = 18 * 5               # Maximum obstacle count

    delta_q_max: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([1, 1, 1, 1, 1, 1], dtype=np.float64) * 5)
    )
    Dq_max: np.ndarray = field(
        default_factory=lambda: np.pi * np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
    )
    DDq_max: np.ndarray = field(
        default_factory=lambda: 12.7 * np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
    )

    lambda_pos: float = 1.0e-2                # Weight on position tracking
    lambda_vel: float = 1.0e-2                # Weight on velocity tracking
    lambda_scaling: float = 1.0e3            # Weight on nominal speed preservation
    lambda_acc: float = 0.0                   # Weight on joint accelerations
    DDtrajectory_time_max: float = 1.0       # Max time scaling acceleration

    delta_unfeasible: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 500
    )
    prefix: str = "ur10e_"
    tool_frame: str = "tool0"
    elbow_frame: str = "forearm_link"

    def __str__(self) -> str:
        """Return formatted string representation of controller configuration."""
        def fmt_arr(arr: np.ndarray) -> str:
            return np.array2string(arr, precision=4, suppress_small=True, separator=", ")

        return (
            "ControllerConfig:\n"
            "  -- Core Parameters --\n"
            f"  Tc              : {self.Tc}\n"
            f"  C               : {self.C}\n"
            f"  Tr              : {self.Tr}\n"
            f"  a_s             : {self.a_s}\n"
            f"  max_obstacles   : {self.max_obstacles}\n\n"
            "  -- Kinematic Limits --\n"
            f"  delta_q_max     : {fmt_arr(self.delta_q_max)}\n"
            f"  Dq_max          : {fmt_arr(self.Dq_max)}\n"
            f"  DDq_max         : {fmt_arr(self.DDq_max)}\n"
            f"  delta_unfeasible: {fmt_arr(self.delta_unfeasible)}\n\n"
            "  -- Weights & Trajectory --\n"
            f"  lambda_pos      : {self.lambda_pos}\n"
            f"  lambda_vel      : {self.lambda_vel}\n"
            f"  lambda_scaling  : {self.lambda_scaling}\n"
            f"  lambda_acc      : {self.lambda_acc}\n"
            f"  gamma           : {self.gamma}\n"
            f"  DDtraj_time_max : {self.DDtrajectory_time_max}\n\n"
            "  -- Robot Frames --\n"
            f"  prefix          : '{self.prefix}'\n"
            f"  tool_frame      : '{self.tool_frame}'\n"
            f"  elbow_frame     : '{self.elbow_frame}'"
        )


class BCFOptimalController:
    """Optimal task controller with Control Barrier Functions (CBF) formulated as a Quadratic Program (QP)."""

    def __init__(
        self,
        model_wrapper: Any,
        cfg: ControllerConfig,
        useCbf: bool = True,
        keypoint_to_log: int = 7,
    ) -> None:
        self.cfg = cfg
        self.model_wrapper = model_wrapper
        self.model = self.model_wrapper.model
        self.data = self.model.createData()

        self.useCbf = useCbf
        tool_name = cfg.tool_frame if cfg.tool_frame.startswith(cfg.prefix) else cfg.prefix + cfg.tool_frame
        elbow_name = cfg.elbow_frame if cfg.elbow_frame.startswith(cfg.prefix) else cfg.prefix + cfg.elbow_frame
        self.tool_frame_id = self.model.getFrameId(tool_name)
        self.elbow_frame_id = self.model.getFrameId(elbow_name)
        self.frames_ids = [self.tool_frame_id, self.elbow_frame_id]

        # Numba prebuilt integrator blocks
        self.FreePos, self.ForcedPos, self.FreeVel, self.ForcedVel = build_free_forced_one_step(
            self.cfg.Tc, self.model.nq
        )

        # QP objective blocks
        nq = self.model.nq
        I = np.eye(nq, dtype=np.float64)
        self.P_pos = np.zeros((nq + 1, nq + 1), dtype=np.float64)
        self.P_pos[:nq, :nq] = 0.25 * (self.cfg.Tc ** 4) * I

        self.P_vel = np.zeros((nq + 1, nq + 1), dtype=np.float64)
        self.P_vel[:nq, :nq] = (self.cfg.Tc ** 2) * I
        self.P_scaling = np.zeros((nq + 1, nq + 1), dtype=np.float64)
        self.P_scaling[-1, -1] = self.cfg.Tc ** 2
        self.P_acc = np.zeros((nq + 1, nq + 1), dtype=np.float64)
        self.P_acc[:nq, :nq] = I

        self.Punfeasible = np.zeros((nq + 1, nq + 1), dtype=np.float64)
        self.Punfeasible[:nq, :nq] = (self.cfg.Tc ** 2) * I
        self.Punfeasible[-1, -1] = self.cfg.Tc ** 2

        # Linear objective vectors
        self.b_pos = np.zeros(nq + 1, dtype=np.float64)
        self.b_vel = np.zeros(nq + 1, dtype=np.float64)
        self.b_scaling = np.zeros(nq + 1, dtype=np.float64)
        self.b_acc = np.zeros(nq + 1, dtype=np.float64)
        self.bunfeasible = np.zeros(nq + 1, dtype=np.float64)

        # Constraint matrices upper bounds
        if useCbf:
            self.n_constraints = 3 + 2 * 3 * nq + cfg.max_obstacles * len(self.frames_ids)
        else:
            self.n_constraints = 3 + 2 * 3 * nq
        self.A = np.zeros((self.n_constraints, nq + 1), dtype=np.float64)
        self.c = np.zeros(self.n_constraints, dtype=np.float64)

        # State initialization
        self.reset_state(np.zeros(nq))

        self.delta_q_max = np.copy(cfg.delta_q_max)
        self.unfeasible_cnt = "FEASIBLE"
        self.ref_scaling = 1.0
        self.qp_scaling = self.ref_scaling
        self.check_delta = False

        self.keypoint_to_log = keypoint_to_log
        self.og_lamda_pos = self.cfg.lambda_pos

    def set_ref_scaling(self, scaling: float) -> None:
        """Set reference velocity scaling clamped between 0 and 1."""
        if scaling >= 1.0:
            self.ref_scaling = 1.0
        elif scaling <= 0.0:
            self.ref_scaling = 0.0
        else:
            self.ref_scaling = float(scaling)

    def reset_state(self, q0: np.ndarray, dq0: Optional[np.ndarray] = None) -> None:
        """Reset internal integrator states."""
        self.q = np.array(q0, dtype=np.float64).copy()
        self.dq = (
            np.zeros_like(self.q)
            if dq0 is None
            else np.array(dq0, dtype=np.float64).copy()
        )
        self.ddq = np.zeros_like(self.q)
        self.t = 0.0
        self.trajectory_time = 0.0
        self.Dtrajectory_time = 1.0
        self.DDtrajectory_time = 0.0

    def step(
        self,
        obs_pos: np.ndarray,
        obs_vel: np.ndarray,
        obs_acc: np.ndarray,
        nominal_q: np.ndarray,
        nominal_Dq: np.ndarray,
        nominal_DDq: np.ndarray,
    ) -> Dict[str, Any]:
        """Perform one discrete control loop iteration."""
        test_unfeasible = 0
        cfg, Tc, nq = self.cfg, self.cfg.Tc, self.model.nq

        nominal_q = np.asarray(nominal_q, dtype=np.float64)
        nominal_Dq = np.asarray(nominal_Dq, dtype=np.float64)
        trajectory_err = float(np.linalg.norm(self.q - nominal_q))

        # Nominal pose for diagnostics
        pin.framesForwardKinematics(self.model, self.data, nominal_q)
        Tbt_nominal = self.data.oMf[self.tool_frame_id].copy()

        # Kinematic state and frame Jacobians
        pin.computeForwardKinematicsDerivatives(self.model, self.data, self.q, self.dq, self.ddq)

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
            Jlins[i, :, :] = J[:3, :]
            dJlins[i, :, :] = dJ[:3, :]

        # Velocity scaling calculation
        if not self.check_delta:
            if not self.useCbf:
                ref_scaling = compute_velocity_scaling_for_human_proximity(
                    model=self.model,
                    data=self.data,
                    q=self.q,
                    dq=self.dq,
                    ddq=self.ddq,
                    tool_frame_ids=self.frames_ids,
                    human_positions_world=obs_pos,
                )
                self.set_ref_scaling(ref_scaling)
            self.qp_scaling = self.ref_scaling
        else:
            self.qp_scaling = 0.0

        # Assemble QP in-place using Numba kernel
        row, h_min, d_min, vr_min, vh_min, htest, dtest, i_h, i_d = assemble_qp_inplace(
            self.P_vel, self.b_pos, self.b_vel, self.b_scaling,
            self.A, self.c,
            self.FreePos, self.ForcedPos, self.FreeVel, self.ForcedVel,
            self.q, self.dq,
            nominal_q, nominal_Dq,
            self.Dtrajectory_time, Tc,
            cfg.Dq_max, cfg.DDq_max, self.delta_q_max,
            frames_p, frames_v, Jlins, dJlins, obs_pos, obs_vel, obs_acc,
            cfg.Tr, cfg.a_s, cfg.C, cfg.gamma, cfg.DDtrajectory_time_max, 1e-12, self.qp_scaling, self.useCbf,
            self.keypoint_to_log,
        )

        if row < self.n_constraints:
            self.A[row:, :].fill(0.0)
            self.c[row:].fill(-1.0)

        # Allow subclasses to modulate parameters dynamically
        self.update_parameters(h_min, d_min, vr_min)

        # Build QP problem
        P = (
            cfg.lambda_pos * self.P_pos
            + cfg.lambda_vel * self.P_vel
            + cfg.lambda_scaling * self.P_scaling
            + cfg.lambda_acc * self.P_acc
        )
        b = (
            cfg.lambda_pos * self.b_pos
            + cfg.lambda_vel * self.b_vel
            + cfg.lambda_scaling * self.b_scaling
            + cfg.lambda_acc * self.b_acc
        )

        P = np.ascontiguousarray(P, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        A = np.ascontiguousarray(self.A, dtype=np.float64)
        c = np.ascontiguousarray(self.c, dtype=np.float64)

        try:
            u, *_ = quadprog.solve_qp(P, b, A.T, c, 0)
        except ValueError as err:
            if "constraints are inconsistent" in str(err):
                # QP Infeasible fallback mode
                self.bunfeasible[:-1] = -Tc * self.dq
                self.bunfeasible[-1] = -Tc * self.Dtrajectory_time
                u, *_ = quadprog.solve_qp(
                    self.Punfeasible, self.bunfeasible,
                    A[(3 + nq * 4):(3 + nq * 6), :].T,
                    c[(3 + nq * 4):(3 + nq * 6)],
                )
                test_unfeasible = 1
                self.unfeasible_cnt = "UNFEASIBLE"
                self.qp_scaling = 0.0
                self.cfg.lambda_pos = self.og_lamda_pos * 1000.0
                self.check_delta = True
            else:
                raise

        # Integrate joint state and time scaling
        self.ddq = u[:-1]
        self.DDtrajectory_time = u[-1]

        self.q = self.q + self.dq * Tc + 0.5 * self.ddq * (Tc ** 2)
        self.dq = self.dq + self.ddq * Tc
        self.trajectory_time += self.Dtrajectory_time * Tc + 0.5 * self.DDtrajectory_time * (Tc ** 2)
        self.Dtrajectory_time += self.DDtrajectory_time * Tc
        self.t += Tc

        pin.framesForwardKinematics(self.model, self.data, self.q)
        Tbt_new = self.data.oMf[self.tool_frame_id]
        frames_p[-1, :] = Tbt_new.translation
        twist = pin.getFrameVelocity(self.model, self.data, self.tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        frames_v[-1, :] = twist.linear
        expected_trj_err = np.abs(self.q - nominal_q)

        if test_unfeasible == 1:
            for i in range(cfg.delta_q_max.shape[0]):
                if expected_trj_err[i] > self.delta_q_max[i]:
                    self.delta_q_max[i] = expected_trj_err[i]
        elif self.check_delta:
            count_dev = 0
            for i in range(nq):
                if expected_trj_err[i] <= cfg.delta_q_max[i]:
                    self.delta_q_max[i] = np.copy(cfg.delta_q_max[i])
                    count_dev += 1
            if count_dev == nq:
                self.check_delta = False
                self.cfg.lambda_pos = self.og_lamda_pos
                self.unfeasible_cnt = "FEASIBLE"
            else:
                self.unfeasible_cnt = "RECOVERING"

        return {
            "h_min": float(h_min),
            "d_min": float(d_min),
            "vr_min": float(vr_min),
            "vh_min": float(vh_min),
            "end_effector_pos": frames_p[-1, :].copy(),
            "end_effector_vel": frames_v[-1, :].copy(),
            "trajectory_error": float(trajectory_err),
            "Tbt_nominal": Tbt_nominal,
            "obs_pos": obs_pos,
            "q": self.q.copy(),
            "dq": self.dq.copy(),
            "ddq": self.ddq.copy(),
            "trajectory_time": float(self.trajectory_time),
            "Dtrajectory_time": float(self.Dtrajectory_time),
            "DDtrajectory_time": float(self.DDtrajectory_time),
            "unfeasible_cnt": self.unfeasible_cnt,
            "unfeasible": self.unfeasible_cnt,
        }

    def update_parameters(self, h: float, d: float, v_rel: float) -> None:
        """Hook for derived controllers to update parameter lambdas dynamically."""
        pass
