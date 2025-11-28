
from dataclasses import dataclass, field
import numpy as np
import quadprog
import pinocchio as pin

from numba_kernels import build_free_forced_one_step, assemble_qp_inplace


@dataclass
class ControllerConfig:
    Tc: float = 2e-3
    C: float = 0.25
    Tr: float = 0.5
    a_s: float = 4.5
    gamma: float = 5.0
    max_obstacles: int = 18 * 5

    # ✅ use default_factory for mutable defaults
    delta_q_max: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.array([1,1,1,1,1,1], dtype=np.float64) * 5)
    )
    Dq_max: np.ndarray = field(
        default_factory=lambda: np.pi * np.array([1,1,1,1,1,1], dtype=np.float64) * np.pi
    )
    DDq_max: np.ndarray = field(
        default_factory=lambda: np.pi * np.array([1,1,1,1,1,1], dtype=np.float64) * np.pi*5.0
    )

    lambda1: float = 1.0e-2
    lambda2: float = 1.0e-2
    lambda3: float = 1.0e3
    lambda4: float = 0.0
    DDtrajectory_time_max: float = 1.0

    delta_unfeasible: np.ndarray = field(
        default_factory=lambda: np.array([1.0,1.0,1.0,1.0,1.0,1.0]) * 500)
    prefix: str = "ur10e_"
    tool_frame: str = "tool0"
    elbow_frame: str = "forearm_link"

class BCFOptimalController:
    """
    Minimal controller: only robot model, state, and QP assembly/solve.
    External code must provide obstacle states and nominal trajectory at each step.
    """
    def __init__(self, model_wrapper, cfg: ControllerConfig):
        self.cfg = cfg
        self.model_wrapper = model_wrapper
        self.model = self.model_wrapper.model
        self.data = self.model.createData()

        # frames (IDs)
        self.tool_frame_id = self.model.getFrameId(cfg.prefix + cfg.tool_frame)
        self.elbow_frame_id = self.model.getFrameId(cfg.prefix + cfg.elbow_frame)
        self.frames_ids = [self.elbow_frame_id, self.tool_frame_id]

        # NUMBA-prebuilt blocks
        self.FreePos, self.ForcedPos, self.FreeVel, self.ForcedVel = \
            build_free_forced_one_step(self.cfg.Tc, self.model.nq)

        # QP static blocks (float64)
        nq = self.model.nq
        I = np.eye(nq, dtype=np.float64)
        self.P1 = np.zeros((nq + 1, nq + 1), dtype=np.float64)
        self.P1[:nq, :nq] = 0.25 * (self.cfg.Tc ** 4) * I

        self.P2 = np.zeros((nq + 1, nq + 1), dtype=np.float64)
        self.P2[:nq, :nq] =(self.cfg.Tc ** 2) * I
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
        self.n_constraints = 3 + 2 * 3 * nq + cfg.max_obstacles * len(self.frames_ids)
        self.A = np.zeros((self.n_constraints, nq + 1), dtype=np.float64)
        self.c = np.zeros(self.n_constraints, dtype=np.float64)

        # state
        self.reset_state(np.zeros(nq))
        
        

    # ---------------------- state ----------------------
    def reset_state(self, q0, dq0=None):
        self.q  = np.array(q0, dtype=np.float64).copy()
        self.dq = (np.zeros_like(self.q) if dq0 is None else np.array(dq0, dtype=np.float64).copy())
        self.ddq = np.zeros_like(self.q)
        self.t = 0.0
        self.trajectory_time = 0.0
        self.Dtrajectory_time = 1.0
        self.DDtrajectory_time = 0.0



    # ---------------------- control step ----------------------
    def step(self, obs_pos, obs_vel, obs_acc,
             nominal_q, nominal_Dq, nominal_DDq):
        """
        Perform one control iteration.
        Inputs:
            obs_pos, obs_vel, obs_acc: (nO,3) float64 arrays
            nominal_q, nominal_Dq, nominal_DDq: vectors for current nominal trajectory
        Returns a dict with diagnostics plus the updated state variables.
        """
        cfg, Tc, nq = self.cfg, self.cfg.Tc, self.model.nq

        nominal_q = np.asarray(nominal_q, dtype=np.float64)
        nominal_Dq = np.asarray(nominal_Dq, dtype=np.float64)

        trajectory_err = float(np.linalg.norm(self.q - nominal_q))

        # Nominal pose for viz/diagnostics
        pin.framesForwardKinematics(self.model, self.data, nominal_q)
        Tbt_nominal = self.data.oMf[self.tool_frame_id].copy()

        # Derivatives for current state (CBF jacobians)
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
            Jlins[i, :, :]  = J[:3, :]
            dJlins[i, :, :] = dJ[:3, :]

        # NUMBA: assemble constraints + objective partials
        row, h_min, d_min, vrel_min = assemble_qp_inplace(
            self.P2, self.b1, self.b2, self.b3,
            self.A, self.c,
            self.FreePos, self.ForcedPos, self.FreeVel, self.ForcedVel,
            self.q, self.dq,
            nominal_q, nominal_Dq,
            self.Dtrajectory_time, Tc,
            cfg.Dq_max, cfg.DDq_max, cfg.delta_q_max,
            frames_p, frames_v, Jlins, dJlins, obs_pos, obs_vel, obs_acc,
            cfg.Tr, cfg.a_s, cfg.C, cfg.gamma,cfg.DDtrajectory_time_max, 1e-12
        )

        # Fix DDtrajectory_time bound
        #self.c[2] = -cfg.DDtrajectory_time_max

        if row < self.n_constraints:
            self.A[row:, :].fill(0.0)
            self.c[row:].fill(-1.0)

        # Dense QP matrices/vectors
        P = (cfg.lambda1 * self.P1 +
             cfg.lambda2 * self.P2 +
             cfg.lambda3 * self.P3 +
             cfg.lambda4 * self.P4)
        b = (cfg.lambda1 * self.b1 +
             cfg.lambda2 * self.b2 +
             cfg.lambda3 * self.b3 +
             cfg.lambda4 * self.b4)

        P = np.ascontiguousarray(P, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        A = np.ascontiguousarray(self.A, dtype=np.float64)
        c = np.ascontiguousarray(self.c, dtype=np.float64)

        # Solve QP
        try:
            u, *_ = quadprog.solve_qp(P, b, A.T, c, 0)
        except ValueError as err:
            if "constraints are inconsistent" in str(err):
                # if h_min>-10:
                # print(f"UNFEASIBLE but h={h_min}")
                # print(f"unfeasible     q = {(np.abs(self.q - nominal_q) > self.cfg.delta_q_max).T}")
                # print(f"unfeasible    dq = {(np.abs(self.dq) > self.cfg.Dq_max).T}")
                # print(f"unfeasible  Dtrj = {self.Dtrajectory_time<0 or self.Dtrajectory_time>1}. Dtrj={self.Dtrajectory_time}")

                # A_unfeasible = np.zeros((3 + 6*nq, nq + 1), dtype=np.float64)
                # c_unfeasible = np.zeros(3 + 6*nq, dtype=np.float64)
                # A_unfeasible[:(3 + 4*nq), :] = A[:(3 + 4*nq), :]
                # c_unfeasible[:(3 + 4*nq)] = c[:(3 + 4*nq)]   
                # row = 3 + 4 * nq
                # # I*u <= ddq + delta_unfeasible
                # for i in range(nq):
                #     for j in range(nq):
                #         A_unfeasible[row + i, j] = -1.0 if i == j else 0.0
                #     c_unfeasible[row + i] = -self.ddq[i] - self.cfg.delta_unfeasible[i]
                # row += nq

                # # -I*u <= delta_unfeasible - ddq
                # for i in range(nq):
                #     for j in range(nq):
                #         A_unfeasible[row + i, j] = 1.0 if i == j else 0.0
                #     c_unfeasible[row + i] = self.ddq[i] - self.cfg.delta_unfeasible[i]
                # row += nq



                self.bunfeasible[:-1] = -Tc * self.dq
                self.bunfeasible[-1]  = -Tc * self.Dtrajectory_time
                u, *_ = quadprog.solve_qp(
                    self.Punfeasible, self.bunfeasible,
                    A[:(3 + nq * 4), :].T,
                    c[:(3 + nq * 4)]
                )
                # u, *_ = quadprog.solve_qp(
                #     self.Punfeasible, self.bunfeasible,
                #     A_unfeasible.T,
                #     c_unfeasible
                # )
            else:
                raise

        # Integrate
        self.ddq = u[:-1]
        self.DDtrajectory_time = u[-1]

        self.q  = self.q + self.dq * Tc + 0.5 * self.ddq * (Tc ** 2)
        self.dq = self.dq + self.ddq * Tc
        self.trajectory_time += self.Dtrajectory_time * Tc + 0.5 * self.DDtrajectory_time * Tc ** 2
        self.Dtrajectory_time += self.DDtrajectory_time * Tc
        self.t += Tc

        pin.framesForwardKinematics(self.model, self.data, self.q)
        Tbt_new = self.data.oMf[self.tool_frame_id]
        frames_p[-1, :] = Tbt_new.translation
        twist = pin.getFrameVelocity(self.model, self.data, self.tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        frames_v[-1, :] = twist.linear
        return {
            "h_min": float(h_min),
            "d_min": float(d_min),
            "vrel_min": float(vrel_min),
            "end_effector_pos": frames_p[-1, :].copy(),
            "end_effector_vel": frames_v[-1, :].copy(),
            "trajectory_error": float(trajectory_err),
            "Tbt_nominal": Tbt_nominal,
            "obs_pos": obs_pos,
            # state echoes
            "q": self.q.copy(),
            "dq": self.dq.copy(),
            "ddq": self.ddq.copy(),
            "trajectory_time": float(self.trajectory_time),
            "Dtrajectory_time": float(self.Dtrajectory_time),
            "DDtrajectory_time": float(self.DDtrajectory_time),
        }
