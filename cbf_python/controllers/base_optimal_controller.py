"""
Base Control Barrier Function (CBF) optimal task controller and configuration.

This module provides the optimization-based motion controller that simultaneously
regulates geometric trajectory progression, bounds path deviations, and enforces
pairwise Control Barrier Function (CBF) safety constraints in shared human-robot workspaces.
Theoretical Overview:
    The motion control problem is formulated as a Quadratic Program (QP) executed at
    each discrete control cycle (typically 500 Hz). The controller determines joint
    accelerations u_k in R^nq and a path-progression acceleration ddot{xi}_k in R:
        min_{u_k, ddot{xi}_k, eps_k}
            lambda_pos * ||q_{k+1} - q_nom_{k+1}||^2
          + lambda_vel * ||dq_{k+1} - dq_nom_{k+1}||^2
          + lambda_scaling * (dot{xi}_{k+1} - dot{xi}_ref)^2
          + lambda_acc * ||u_k||^2
          + lambda_tube * ||eps_k||^2
    Subject to:
        1. Kinematic joint position limits: q_min <= q_{k+1} <= q_max
        2. Velocity limits: |dq_{k+1}| <= dq_max
        3. Actuator acceleration limits: |u_k| <= ddq_max
        4. Path progression rate limits: 0 <= dot{xi}_{k+1} <= 1, ddot{xi}_k <= ddot{xi}_max
        5. Joint-space trajectory tube: |q_{k+1} - q_nom_{k+1}| <= delta_q_max + eps_k, eps_k >= 0
        6. Robust stopping-horizon CBF constraints: A_cbf * u_k >= bound

    Safety Fail-Safe Strategy:
        If solver execution overruns the real-time deadline or encounters unexpected
        numerical anomalies, the controller triggers a deterministic emergency braking
        fallback with maximum deceleration, followed by a monotonic tube recovery sequence.
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

    Tc: float = 2e-3                          # Control loop period [s] (e.g. 500 Hz -> 0.002 s)
    C: float = 0.25                           # Safety separation margin [m] (lumped threshold)
    C_0: float = 0.10                         # Static perception uncertainty threshold [m]
    beta: float = 3.291                       # Gaussian confidence scaling factor (99.9% confidence)
    T_f_bar: float = 0.3                      # Conservative upper bound on robot stopping duration [s]
    sigma_p: float = 0.015                    # Position measurement standard deviation [m]
    sigma_v: float = 0.072                    # Velocity estimation standard deviation [m/s]
    sigma_a: float = 0.196                    # Acceleration estimation standard deviation [m/s^2]
    Tr: float = 0.15                          # System reaction latency [s] (sensing + communication delay)
    a_s: float = 2.5                          # Manipulator maximum braking deceleration capability [m/s^2]
    gamma: float = 5.0                        # Control Barrier Function class-K linear responsiveness gain [1/s]
    max_obstacles: int = 18 * 5               # Maximum monitored obstacle keypoint count

    delta_q_max: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([1, 1, 1, 1, 1, 1], dtype=np.float64) * 5)
    )                                         # Joint-space nominal path tube half-widths [rad]
    Dq_max: np.ndarray = field(
        default_factory=lambda: np.pi * np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
    )                                         # Maximum permissible joint velocities [rad/s]
    DDq_max: np.ndarray = field(
        default_factory=lambda: 12.7 * np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
    )                                         # Maximum admissible joint accelerations [rad/s^2]

    lambda_pos: float = 1.0e-2                # Objective weight on position tracking error
    lambda_vel: float = 1.0e-2                # Objective weight on velocity tracking error
    lambda_scaling: float = 1.0e3            # Objective weight on maintaining nominal path progression
    lambda_acc: float = 0.0                   # Objective weight on joint acceleration norm (actuator effort)
    DDtrajectory_time_max: float = 1.0       # Maximum admissible scaling acceleration ddot{xi}_max [s^-2]

    # Soft-tube slack configuration
    use_soft_tube: bool = True               # Enable slack variable relaxation on trajectory tube
    lambda_tube: float = 1.0e5               # Quadratic penalty weight on path tube slack violations

    delta_unfeasible: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 500
    )
    prefix: str = "ur10e_"
    tool_frame: str = "tool0"
    elbow_frame: str = "forearm_link"

    # Volumetric Capsule Distance Model
    use_capsules: bool = False                # Enable continuous segment-to-segment capsule checking
    robot_capsule_radius: float = 0.08        # Robot link cylindrical capsule radius [m]
    human_capsule_radius: float = 0.08        # Human keypoint/limb sphere/capsule radius [m]

    @property
    def delta_H(self) -> float:
        """
        Maximum anticipated positional error in human motion estimation over stopping horizon.

        Evaluates worst-case Taylor expansion error:
            Delta_H = beta * (sigma_p + sigma_v * T_f_bar + 0.5 * sigma_a * T_f_bar^2)
        """
        return self.beta * (self.sigma_p + self.sigma_v * self.T_f_bar + 0.5 * self.sigma_a * (self.T_f_bar ** 2))

    def compute_missed_cycle_margin(
        self,
        v_r_bar: float = 1.0,
        v_h_bar: float = 1.6,
        a_r_bar: float = 2.5,
        a_h_bar: float = 4.5,
    ) -> float:
        """
        Compute additional distance margin Delta d_miss to absorb a single zero-order hold cycle.
        """
        return (v_r_bar + v_h_bar) * self.Tc + 0.5 * (a_r_bar + a_h_bar) * (self.Tc ** 2)

    def compute_robust_margin(
        self,
        v_r_bar: float = 1.0,
        v_h_bar: float = 1.6,
        a_r_bar: float = 2.5,
        a_h_bar: float = 4.5,
    ) -> float:
        """
        Compute total robust clearance threshold C^rob = C_0 + Delta d_miss.
        """
        margin = self.C_0 if self.C_0 is not None else self.C
        return margin + self.compute_missed_cycle_margin(v_r_bar, v_h_bar, a_r_bar, a_h_bar)

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
            f"  DDtraj_time_max : {self.DDtrajectory_time_max}\n"
            f"  use_soft_tube   : {self.use_soft_tube}\n"
            f"  lambda_tube     : {self.lambda_tube}\n\n"
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
        # Order frames from base to end-effector so that index -1 corresponds to the tool frame
        self.frames_ids = [self.elbow_frame_id, self.tool_frame_id]
        # Volumetric capsule segments: segment (0, 1) spans between elbow and tool
        self.capsule_segments = (
            np.array([[0, 1]], dtype=np.int64) if cfg.use_capsules else np.empty((0, 2), dtype=np.int64)
        )

        nq = self.model.nq
        self.nq = nq

        # Determine decision variable dimension:
        # If soft tube is enabled: decision vector is [ddq (nq), DDtrajectory_time (1), eps (nq)] -> 2*nq + 1
        # If hard tube is enabled: decision vector is [ddq (nq), DDtrajectory_time (1)] -> nq + 1
        self.n_vars = (2 * nq + 1) if cfg.use_soft_tube else (nq + 1)

        # Numba prebuilt discrete-time integrator blocks
        self.FreePos, self.ForcedPos, self.FreeVel, self.ForcedVel = build_free_forced_one_step(
            self.cfg.Tc, nq
        )

        # QP objective Hessian blocks
        I = np.eye(nq, dtype=np.float64)
        self.P_pos = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)
        self.P_pos[:nq, :nq] = 0.25 * (self.cfg.Tc ** 4) * I

        self.P_vel = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)
        self.P_vel[:nq, :nq] = (self.cfg.Tc ** 2) * I

        self.P_scaling = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)
        self.P_scaling[nq, nq] = self.cfg.Tc ** 2

        self.P_acc = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)
        self.P_acc[:nq, :nq] = I

        # Quadratic penalty on soft tube slack: lambda_tube * ||eps||^2
        self.P_tube = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)
        if cfg.use_soft_tube:
            self.P_tube[nq + 1:, nq + 1:] = np.eye(nq, dtype=np.float64)

        # Linear objective gradient vectors
        self.b_pos = np.zeros(self.n_vars, dtype=np.float64)
        self.b_vel = np.zeros(self.n_vars, dtype=np.float64)
        self.b_scaling = np.zeros(self.n_vars, dtype=np.float64)
        self.b_acc = np.zeros(self.n_vars, dtype=np.float64)

        # Joint physical position bounds
        self.q_min = np.copy(self.model.lowerPositionLimit)
        self.q_max = np.copy(self.model.upperPositionLimit)

        # Total constraint rows:
        # Scaling (3) + Position (2*nq) + Tube (2*nq) + Velocity (2*nq) + Acceleration (2*nq)
        # + Slack non-negativity (nq, if enabled) + Pairwise CBFs (max_obstacles * monitored entities)
        base_constraints = 3 + 2 * 4 * nq + (nq if cfg.use_soft_tube else 0)
        n_monitored = max(len(self.capsule_segments), 1) if cfg.use_capsules else len(self.frames_ids)
        if useCbf:
            self.n_constraints = base_constraints + cfg.max_obstacles * n_monitored
        else:
            self.n_constraints = base_constraints

        self.A = np.zeros((self.n_constraints, self.n_vars), dtype=np.float64)
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
        self.slack = np.zeros(nq, dtype=np.float64)

    def set_ref_scaling(self, scaling: float) -> None:
        """Set reference velocity scaling clamped between 0 and 1."""
        if scaling >= 1.0:
            self.ref_scaling = 1.0
        elif scaling <= 0.0:
            self.ref_scaling = 0.0
        else:
            self.ref_scaling = float(scaling)

    def reset_state(self, q0: np.ndarray, dq0: Optional[np.ndarray] = None) -> None:
        """Reset internal integrator states to initial conditions."""
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
        self.u_prev = np.zeros(self.n_vars, dtype=np.float64)
        self.N_miss = 0
        self.slack = np.zeros(self.nq, dtype=np.float64)

    def step(
        self,
        obs_pos: np.ndarray,
        obs_vel: np.ndarray,
        obs_acc: np.ndarray,
        nominal_q: np.ndarray,
        nominal_Dq: np.ndarray,
        nominal_DDq: np.ndarray,
        deadline_missed: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute one discrete control loop iteration.

        Parameters:
            obs_pos: Detected obstacle/human positions in world frame (nO, 3)
            obs_vel: Detected obstacle/human velocities in world frame (nO, 3)
            obs_acc: Detected obstacle/human accelerations in world frame (nO, 3)
            nominal_q: Desired trajectory joint position reference at current step
            nominal_Dq: Desired trajectory joint velocity reference
            nominal_DDq: Desired trajectory joint acceleration reference
            deadline_missed: Boolean flag indicating if solver execution timed out

        Returns:
            Dictionary containing commanded kinematics, safety margins, and diagnostic metrics.
        """
        test_unfeasible = 0
        cfg, Tc, nq = self.cfg, self.cfg.Tc, self.model.nq

        nominal_q = np.asarray(nominal_q, dtype=np.float64)
        nominal_Dq = np.asarray(nominal_Dq, dtype=np.float64)
        trajectory_err = float(np.linalg.norm(self.q - nominal_q))

        # Forward kinematics for nominal pose logging
        pin.framesForwardKinematics(self.model, self.data, nominal_q)
        Tbt_nominal = self.data.oMf[self.tool_frame_id].copy()

        # Compute forward kinematics and frame Jacobians for monitored links
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

        # Velocity scaling evaluation
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
            # During tube recovery, progression along nominal trajectory is suspended
            self.qp_scaling = 0.0
            self.Dtrajectory_time = 0.0

        # Assemble QP constraints and cost parts in-place
        margin_C0 = cfg.C_0 if getattr(cfg, "C_0", None) is not None else cfg.C
        delta_H = getattr(cfg, "delta_H", 0.0)
        sigma_psi = np.array([cfg.sigma_p, 0.0, cfg.sigma_v, cfg.sigma_a], dtype=np.float64)

        row, h_min, d_min, vr_min, vh_min, htest, dtest, i_h, i_d = assemble_qp_inplace(
            self.P_vel, self.b_pos, self.b_vel, self.b_scaling,
            self.A, self.c,
            self.FreePos, self.ForcedPos, self.FreeVel, self.ForcedVel,
            self.q, self.dq,
            nominal_q, nominal_Dq,
            self.Dtrajectory_time, Tc,
            self.q_min, self.q_max,
            cfg.Dq_max, cfg.DDq_max, self.delta_q_max,
            frames_p, frames_v, Jlins, dJlins, obs_pos, obs_vel, obs_acc,
            cfg.Tr, cfg.a_s, margin_C0, cfg.gamma, cfg.DDtrajectory_time_max, 1e-12, self.qp_scaling, self.useCbf,
            self.keypoint_to_log, delta_H,
            sigma_psi, cfg.beta,
            self.capsule_segments, cfg.robot_capsule_radius, cfg.human_capsule_radius,
        )

        if row < self.n_constraints:
            self.A[row:, :].fill(0.0)
            self.c[row:].fill(-1.0)

        # Allow derived classes to modulate parameters dynamically
        self.update_parameters(h_min, d_min, vr_min)

        # Construct composite Hessian matrix and linear gradient vector
        P = (
            cfg.lambda_pos * self.P_pos
            + cfg.lambda_vel * self.P_vel
            + cfg.lambda_scaling * self.P_scaling
            + cfg.lambda_acc * self.P_acc
            + cfg.lambda_tube * self.P_tube
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

        u = np.zeros(self.n_vars, dtype=np.float64)

        # --------------------------------------------------------
        # QP Solution & Safety Fallback Logic
        # --------------------------------------------------------
        if deadline_missed:
            self.N_miss += 1
            if self.N_miss == 1:
                # One missed control cycle: maintain previous valid command (zero-order hold)
                u = self.u_prev.copy()
            else:
                # Multiple consecutive missed deadlines: apply deterministic emergency braking
                u[:nq] = np.clip(-self.dq / Tc, -cfg.DDq_max, cfg.DDq_max)
                u[nq] = np.clip(-self.Dtrajectory_time / Tc, -cfg.DDtrajectory_time_max, cfg.DDtrajectory_time_max)
                if cfg.use_soft_tube:
                    u[nq + 1:] = 0.0
                self.u_prev = u.copy()
            test_unfeasible = 1
            self.unfeasible_cnt = "UNFEASIBLE"
            self.qp_scaling = 0.0
        else:
            try:
                # Solve strictly convex QP via Goldfarb-Idnani dual active-set method
                u, *_ = quadprog.solve_qp(P, b, A.T, c, 0)
                self.N_miss = 0
                self.u_prev = u.copy()
            except ValueError as err:
                if "constraints are inconsistent" in str(err):
                    # Safety fallback mode: command maximum deceleration opposing motion
                    u[:nq] = np.clip(-self.dq / Tc, -cfg.DDq_max, cfg.DDq_max)
                    u[nq] = np.clip(-self.Dtrajectory_time / Tc, -cfg.DDtrajectory_time_max, cfg.DDtrajectory_time_max)
                    if cfg.use_soft_tube:
                        u[nq + 1:] = 0.0
                    self.N_miss = 0
                    self.u_prev = u.copy()
                    test_unfeasible = 1
                    self.unfeasible_cnt = "UNFEASIBLE"
                    self.qp_scaling = 0.0
                else:
                    raise

        # Extract optimized control variables
        self.ddq = u[:nq]
        self.DDtrajectory_time = u[nq]
        if cfg.use_soft_tube:
            self.slack = u[nq + 1:].copy()

        # Discrete-time state integration
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

        # --------------------------------------------------------
        # Tube Recovery Strategy
        # --------------------------------------------------------
        if test_unfeasible == 1:
            # During emergency braking, dynamically expand temporary tube if excursion occurs
            for i in range(nq):
                if expected_trj_err[i] > cfg.delta_q_max[i]:
                    self.delta_q_max[i] = expected_trj_err[i]
                    self.check_delta = True
        elif self.check_delta:
            # Recovery phase: contract temporary tube monotonically back toward nominal bound
            for i in range(nq):
                self.delta_q_max[i] = max(
                    cfg.delta_q_max[i],
                    min(self.delta_q_max[i], expected_trj_err[i]),
                )
            # Recovery terminates once actual tracking error returns within nominal bound
            if np.all(expected_trj_err <= cfg.delta_q_max):
                self.check_delta = False
                self.delta_q_max = np.copy(cfg.delta_q_max)
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
            "slack": self.slack.copy(),
        }

    def update_parameters(self, h: float, d: float, v_rel: float) -> None:
        """
        Hook for derived controllers to update parameter weights dynamically.

        Subclasses (such as PolynomialOptimalController and GaussianController) override
        this method to adapt objective penalties (lambda_pos, lambda_vel, etc.) or CBF
        responsiveness gain (gamma) as a function of the closest obstacle distance and barrier margin.

        Parameters:
            h: Minimum barrier certificate margin across all monitored pairs [m]
            d: Minimum physical distance to closest obstacle [m]
            v_rel: Projected relative approach velocity of the critical pair [m/s]
        """
        pass
