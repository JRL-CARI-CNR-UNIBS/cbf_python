# optimal_bcf_task_controller.py
# Pattern A controller: NUMBA assembles, quadprog solves, viz/log outside step()

from dataclasses import dataclass
import time
import numpy as np
import quadprog
import pinocchio as pin

# your libs

from numba_kernels import (
    build_free_forced_one_step,
    assemble_qp_inplace,
)

@dataclass
class ControllerConfig:
    Tc: float = 2e-3
    duration: float = 30.0

    # Safety / CBF
    C: float = 0.25
    Tr: float = 0.5
    a_s: float = 4.5
    gamma: float = 5.0
    max_obstacles: int = 18 * 5

    # Joint limits
    # (ensure float64 dtype)
    delta_q_max: np.ndarray = np.deg2rad(np.array([1,1,1,1,1,1], dtype=np.float64) * 0.57 * 10)
    Dq_max: np.ndarray      = np.pi * np.array([1,1,1,1,1,1], dtype=np.float64) * 1.0
    DDq_max: np.ndarray     = np.pi * np.array([1,1,1,1,1,1], dtype=np.float64) * 3.0

    # QP weights
    lambda1: float = 1.0e2
    lambda2: float = 1.0
    lambda3: float = 1.0e-1
    lambda4: float = 0.0
    DDtrajectory_time_max: float = 1.0

    # Misc
    use_cbf: bool = True
    use_bridge: bool = False
    ur10e_joint_names: tuple = (
        "ur10e_shoulder_pan_joint",
        "ur10e_shoulder_lift_joint",
        "ur10e_elbow_joint",
        "ur10e_wrist_1_joint",
        "ur10e_wrist_2_joint",
        "ur10e_wrist_3_joint",
    )
    prefix: str = "ur10e_"
    tool_frame: str = "tool0"
    elbow_frame: str = "forearm_link"


class BCFOptimalController:
    def __init__(self, bridge, planner, model_wrapper, cfg: ControllerConfig):
        self.cfg = cfg
        self.bridge = bridge
        self.planner = planner

        # robot model
        self.model_wrapper = model_wrapper
        self.model = self.model_wrapper.model
        self.data = self.model.createData()

        # frames (IDs)
        self.tool_frame_id = self.model.getFrameId(cfg.prefix + cfg.tool_frame)
        self.elbow_frame_id = self.model.getFrameId(cfg.prefix + cfg.elbow_frame)
        self.frames_ids = np.array([self.elbow_frame_id, self.tool_frame_id], dtype=np.int64)

        # NUMBA-prebuilt blocks
        self.FreePos, self.ForcedPos, self.FreeVel, self.ForcedVel = \
            build_free_forced_one_step(self.cfg.Tc, self.model.nq)

        # QP static blocks (float64)
        nq = self.model.nq
        I = np.eye(nq, dtype=np.float64)
        self.P1 = np.zeros((nq + 1, nq + 1), dtype=np.float64)
        self.P1[:nq, :nq] = 0.25 * (self.cfg.Tc ** 4) * I

        self.P2 = np.zeros((nq + 1, nq + 1), dtype=np.float64)
        self.P3 = np.zeros((nq + 1, nq + 1), dtype=np.float64); self.P3[-1, -1] = (self.cfg.Tc ** 2)
        self.P4 = np.zeros((nq + 1, nq + 1), dtype=np.float64); self.P4[:nq, :nq] = I
        self.Punfeasible = np.zeros((nq + 1, nq + 1), dtype=np.float64)
        self.Punfeasible[:nq, :nq] = (self.cfg.Tc ** 2) * I
        self.Punfeasible[-1, -1]   = (self.cfg.Tc ** 2)

        # linear parts
        self.b1 = np.zeros(nq + 1, dtype=np.float64)
        self.b2 = np.zeros(nq + 1, dtype=np.float64)
        self.b3 = np.zeros(nq + 1, dtype=np.float64)
        self.b4 = np.zeros(nq + 1, dtype=np.float64)
        self.bunfeasible = np.zeros(nq + 1, dtype=np.float64)

        # constraints (upper bound)
        self.n_constraints = 3 + 2 * 3 * nq + cfg.max_obstacles * 2  # rough upper bound
        self.A = np.zeros((self.n_constraints, nq + 1), dtype=np.float64)
        self.c = np.zeros(self.n_constraints, dtype=np.float64)

        # logs
        self._reset_logs()

    def reset_state(self, q0, dq0=None):
        self.q  = np.array(q0, dtype=np.float64).copy()
        self.dq = (np.zeros_like(self.q) if dq0 is None else np.array(dq0, dtype=np.float64).copy())
        self.ddq = np.zeros_like(self.q)

        self.t = 0.0
        self.trajectory_time = 0.0
        self.Dtrajectory_time = 1.0
        self.DDtrajectory_time = 0.0

    # ---------------------- control step (no viz/log) ----------------------
    def step(self, loop_start):
        cfg, Tc, nq = self.cfg, self.cfg.Tc, self.model.nq

        # 1) Obstacles & nominal trajectory
        obs_pos, obs_vel, obs_acc = self.bridge.getObstacles()  # expect float64 arrays, shapes (nO,3)
        nominal_q, nominal_Dq, nominal_DDq = self.planner.getMotionLaw(
            self.trajectory_time % self.planner.computeTime()
        )
        nominal_q  = np.asarray(nominal_q, dtype=np.float64)
        nominal_Dq = np.asarray(nominal_Dq, dtype=np.float64)

        trajectory_err = float(np.linalg.norm(self.q - nominal_q))

        # 2) Kinematics for frames (Pinocchio)
        pin.framesForwardKinematics(self.model, self.data, nominal_q)
        Tbt_nominal = self.data.oMf[self.tool_frame_id].copy()

        pin.computeForwardKinematicsDerivatives(self.model, self.data, self.q, self.dq, self.ddq)

        # Prepare arrays for NUMBA CBF append (optional)
        nF = len(self.frames_ids)
        frames_p = np.zeros((nF, 3), dtype=np.float64)
        frames_v = np.zeros((nF, 3), dtype=np.float64)
        Jlins    = np.zeros((nF, 3, nq), dtype=np.float64)
        dJlins   = np.zeros((nF, 3, nq), dtype=np.float64)

        for i, f_id in enumerate(self.frames_ids):
            Tbt = self.data.oMf[f_id]
            frames_p[i, :] = Tbt.translation
            twist = pin.getFrameVelocity(self.model, self.data, f_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            frames_v[i, :] = twist.linear

            J = pin.computeFrameJacobian(self.model, self.data, self.q, f_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            dJ = pin.frameJacobianTimeVariation(self.model, self.data, self.q, self.dq, f_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jlins[i, :, :]  = J[:3, :]
            dJlins[i, :, :] = dJ[:3, :]

        # 3) NUMBA: assemble constraints + objective partials (in-place)
        # NOTE: assemble_qp_inplace includes a placeholder for the DDtraj limit row;
        # we’ll fix the RHS right after.
        row, h_min = assemble_qp_inplace(
            self.P2, self.b1, self.b2, self.b3,
            self.A, self.c,
            self.FreePos, self.ForcedPos, self.FreeVel, self.ForcedVel,
            self.q, self.dq,
            nominal_q, nominal_Dq,
            self.Dtrajectory_time, Tc,
            cfg.Dq_max, cfg.DDq_max, cfg.delta_q_max,
            frames_p, frames_v, Jlins, dJlins,
            obs_pos, obs_vel, obs_acc,
            cfg.Tr, cfg.a_s, cfg.C, cfg.gamma, 1e-12
        )

        # Fix the 3rd constraint RHS to the configured DDtrajectory_time_max
        # The first three rows are scaling rows in this order.
        self.c[2] = -cfg.DDtrajectory_time_max

        # Zero-fill any unused rows (not strictly necessary for quadprog)
        if row < self.n_constraints:
            self.A[row:, :].fill(0.0)
            self.c[row:].fill(-1.0)

        # 4) Form final (dense) QP terms
        P = (cfg.lambda1 * self.P1 +
             cfg.lambda2 * self.P2 +
             cfg.lambda3 * self.P3 +
             cfg.lambda4 * self.P4)
        b = (cfg.lambda1 * self.b1 +
             cfg.lambda2 * self.b2 +
             cfg.lambda3 * self.b3 +
             cfg.lambda4 * self.b4)

        # Ensure dtype/contiguity for quadprog
        P = np.ascontiguousarray(P, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        A = np.ascontiguousarray(self.A, dtype=np.float64)
        c = np.ascontiguousarray(self.c, dtype=np.float64)

        # 5) Solve QP (Python)
        try:
            u, *_ = quadprog.solve_qp(P, b, A.T, c, 0)
        except ValueError as err:
            if "constraints are inconsistent" in str(err):
                self.bunfeasible[:-1] = -Tc * self.dq
                self.bunfeasible[-1]  = -Tc * self.Dtrajectory_time
                u, *_ = quadprog.solve_qp(
                    self.Punfeasible, self.bunfeasible,
                    A[:(2 + nq * 3), :].T,
                    c[:(2 + nq * 3)]
                )
            else:
                raise

        # 6) Integrate (simple, here in Python; can be numba’d if you prefer)
        self.ddq = u[:-1]
        self.DDtrajectory_time = u[-1]

        self.q  = self.q + self.dq * Tc + 0.5 * self.ddq * (Tc ** 2)
        self.dq = self.dq + self.ddq * Tc
        self.trajectory_time += self.Dtrajectory_time * Tc + 0.5 * self.DDtrajectory_time * Tc ** 2
        self.Dtrajectory_time += self.DDtrajectory_time * Tc
        self.t += Tc

        if self.cfg.use_bridge:
            self.bridge.sendCommand(self.q)

        # Return diagnostics only (no I/O here)
        return {
            "h_min": float(h_min),
            "trajectory_error": float(trajectory_err),
            "Tbt_nominal": Tbt_nominal,
            "obs_pos": obs_pos,
            "elapsed": float(time.perf_counter() - loop_start),
        }

    # ---------------------- publish / log outside step ----------------------
    def publish_and_log(self, d):
        viz_text = (
            f"h = {d['h_min']:.2f} m, "
            f"scaling {self.Dtrajectory_time:4.3f}, "
            f"trajectory_error = {d['trajectory_error']:.2f}"
        )
        # self.renderer.push_state(self.q, d["Tbt_nominal"], d["obs_pos"], viz_text)
        self._log(d["elapsed"], d["h_min"], d["trajectory_error"])

    def run(self, duration_s=None):
        duration = self.cfg.duration if duration_s is None else duration_s
        Tc = self.cfg.Tc

        while self.t < duration:
            loop_start = time.perf_counter()
            d = self.step(loop_start)
            self.publish_and_log(d)

            elapsed = time.perf_counter() - loop_start
            rest = Tc - elapsed
            if rest > 0:
                time.sleep(rest)
        self._finalize()

    # ---------------------- utils ----------------------
    def _reset_logs(self):
        self.cycles = 0
        self.ct = []
        self.h_log = []
        self.trj_error_log = []
        self.scaling_log = []

    def _log(self, elapsed, h_min, trajectory_err):
        self.cycles += 1
        self.ct.append(min(50e-3, elapsed))
        self.h_log.append(h_min)
        self.trj_error_log.append(trajectory_err)
        self.scaling_log.append(self.Dtrajectory_time)

    def _finalize(self):
        pass  # your plotting/summary here
