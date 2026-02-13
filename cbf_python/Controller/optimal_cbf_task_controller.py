
from dataclasses import dataclass, field
import numpy as np
import quadprog
import pinocchio as pin

from Controller.Numba_scripts.numba_kernels import build_free_forced_one_step, assemble_qp_inplace
from Controller import compute_velocity_scaling_for_human_proximity as ext_scaling


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

    lambda_pos: float = 1.0e-2
    lambda_vel: float = 1.0e-2
    lambda_scaling: float = 1.0e3
    lambda_acc: float = 0.0
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
    def __init__(self, model_wrapper, cfg: ControllerConfig, useCbf, keypoint_to_log = 7):
        self.cfg = cfg
        self.model_wrapper = model_wrapper
        self.model = self.model_wrapper.model
        self.data = self.model.createData()

        self.useCbf = useCbf
        # frames (IDs)
        self.tool_frame_id = self.model.getFrameId(cfg.prefix + cfg.tool_frame)
        self.elbow_frame_id = self.model.getFrameId(cfg.prefix + cfg.elbow_frame)
        self.frames_ids = [ self.tool_frame_id, self.elbow_frame_id]

        # NUMBA-prebuilt blocks
        self.FreePos, self.ForcedPos, self.FreeVel, self.ForcedVel = \
            build_free_forced_one_step(self.cfg.Tc, self.model.nq)

        # QP static blocks (float64)
        nq = self.model.nq
        I = np.eye(nq, dtype=np.float64)
        self.P_pos = np.zeros((nq + 1, nq + 1), dtype=np.float64)
        self.P_pos[:nq, :nq] = 0.25 * (self.cfg.Tc ** 4) * I

        self.P_vel = np.zeros((nq + 1, nq + 1), dtype=np.float64)
        self.P_vel[:nq, :nq] =(self.cfg.Tc ** 2) * I
        self.P_scaling = np.zeros((nq + 1, nq + 1), dtype=np.float64); self.P_scaling[-1, -1] = (self.cfg.Tc ** 2)
        self.P_acc = np.zeros((nq + 1, nq + 1), dtype=np.float64); self.P_acc[:nq, :nq] = I
        self.Punfeasible = np.zeros((nq + 1, nq + 1), dtype=np.float64)
        self.Punfeasible[:nq, :nq] = (self.cfg.Tc ** 2) * I
        self.Punfeasible[-1, -1]   = (self.cfg.Tc ** 2)

        # linear parts
        self.b_pos = np.zeros(nq + 1, dtype=np.float64)
        self.b_vel = np.zeros(nq + 1, dtype=np.float64)
        self.b_scaling = np.zeros(nq + 1, dtype=np.float64)
        self.b_acc = np.zeros(nq + 1, dtype=np.float64)
        self.bunfeasible = np.zeros(nq + 1, dtype=np.float64)

        # constraints (upper bound)
        if useCbf:
            self.n_constraints = 3 + 2 * 3 * nq + cfg.max_obstacles * len(self.frames_ids)
        else:
            self.n_constraints = 3 + 2 * 3 * nq
        self.A = np.zeros((self.n_constraints, nq + 1), dtype=np.float64)
        self.c = np.zeros(self.n_constraints, dtype=np.float64)

        # state
        self.reset_state(np.zeros(nq))

        # delta_q_max
        self.delta_q_max = np.copy(cfg.delta_q_max)
        self.unfeasible_cnt = "FEASIBLE"
        self.ref_scaling = 1.0
        self.qp_scaling = self.ref_scaling
        # print(f"DELTA_Q_MAX: {self.delta_q_max}")
        self.check_delta = False

        # keypoint to log
        self.keypoint_to_log = keypoint_to_log

    def set_ref_scaling(self, scaling):
        if scaling >= 1.0 :
            self.ref_scaling = 1.0
        elif scaling <= 0.0:
            self.ref_scaling = 0.0
        else:
            self.ref_scaling = scaling


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
        test_unfeasible = 0

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

        # reference scaling computing. If check delta flag is True, it means the controller is in recovery phase after
        # an unfeaible state, so qp_scaling shoud be null. If not, qp_scaling should track the reference scaling attribute.
        if not self.check_delta:
            if not self.useCbf:
                ref_scaling = ext_scaling.compute_velocity_scaling_for_human_proximity(
                    model=self.model.copy(), data=self.data.copy(),
                    q=self.q,
                    dq=self.dq,
                    ddq=self.ddq,
                    tool_frame_ids= self.frames_ids,
                    human_positions_world = obs_pos,
                )
                # print("Reference scaling: ", ref_scaling)
                self.set_ref_scaling(ref_scaling)
                # print("reference scaling attribute: ", self.ref_scaling)
            self.qp_scaling = self.ref_scaling
        else:
            self.qp_scaling = 0.0
        # NUMBA: assemble constraints + objective partials
        row, h_min, d_min, vr_min, vh_min, htest, dtest, i_h, i_d = assemble_qp_inplace(
            self.P_vel, self.b_pos, self.b_vel, self.b_scaling,
            self.A, self.c,
            self.FreePos, self.ForcedPos, self.FreeVel, self.ForcedVel,
            self.q, self.dq,
            nominal_q, nominal_Dq,
            self.Dtrajectory_time, Tc,
            cfg.Dq_max, cfg.DDq_max, self.delta_q_max,
            frames_p, frames_v, Jlins, dJlins, obs_pos, obs_vel, obs_acc,
            cfg.Tr, cfg.a_s, cfg.C, cfg.gamma,cfg.DDtrajectory_time_max, 1e-12, self.qp_scaling, self.useCbf,
            self.keypoint_to_log
        )

        # print(f"h_min: {htest}, on keypoint no: {i_h}")
        # print(f"d_min: {dtest}, on keypoint no: {i_d}")

        # Fix DDtrajectory_time bound
        #self.c[2] = -cfg.DDtrajectory_time_max

        if row < self.n_constraints:
            self.A[row:, :].fill(0.0)
            self.c[row:].fill(-1.0)

            
        self.update_parameters(h_min)

        # print (f"UPDATED LAMBDAS: POS: {self.cfg.lambda_pos}, VEL: {self.cfg.lambda_vel}, SCALING: {self.cfg.lambda_scaling}, ACC: {self.cfg.lambda_acc}, GAMMA: {self.cfg.gamma}, DELTA_Q_MAX: {self.cfg.delta_q_max}")
        # Dense QP matrices/vectors
        P = (cfg.lambda_pos * self.P_pos +
             cfg.lambda_vel * self.P_vel +
             cfg.lambda_scaling * self.P_scaling +
             cfg.lambda_acc * self.P_acc)
        b = (cfg.lambda_pos * self.b_pos +
             cfg.lambda_vel * self.b_vel +
             cfg.lambda_scaling * self.b_scaling +
             cfg.lambda_acc * self.b_acc)

        P = np.ascontiguousarray(P, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        A = np.ascontiguousarray(self.A, dtype=np.float64)
        c = np.ascontiguousarray(self.c, dtype=np.float64)

        # Solve QP
        try:
            u, *_ = quadprog.solve_qp(P, b, A.T, c, 0)
        except ValueError as err:
            if "constraints are inconsistent" in str(err):
                self.bunfeasible[:-1] = -Tc * self.dq
                self.bunfeasible[-1]  = -Tc * self.Dtrajectory_time
                u, *_ = quadprog.solve_qp(
                    self.Punfeasible, self.bunfeasible,
                    A[(3 + nq * 4):(3 + nq*6), :].T,
                    c[(3 + nq * 4):(3 + nq * 6)]
                )
                # u, *_ = quadprog.solve_qp(
                #     self.Punfeasible, self.bunfeasible,
                #     A[:(3 + nq * 4), :].T,
                #     c[:(3 + nq * 4)]
                # )
                test_unfeasible = 1
                self.unfeasible_cnt = "UNFEASIBLE"
                self.qp_scaling = 0.0
                self.check_delta = True
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
        expected_trj_err = abs(self.q - nominal_q)

        if test_unfeasible == 1:
            for i in range(cfg.delta_q_max.shape[0]):
                if expected_trj_err[i] > self.delta_q_max[i]:
                    self.delta_q_max[i] = expected_trj_err[i]
            # print(f"PROBLEM IS UNFEASIBLE, trajectory error: {expected_trj_err}")
            # print(f"NEW DELTA: {self.delta_q_max}")
            # print(f"INITIAL Q MAX: {cfg.delta_q_max}")
        elif self.check_delta:
            count_dev = 0
            for i in range(nq):
                if expected_trj_err[i] <= cfg.delta_q_max[i]:
                    self.delta_q_max[i] = np.copy(cfg.delta_q_max[i])
                    count_dev += 1
                    # print(f"COUNT_DEV: {count_dev}, i: {i}")
            if count_dev == nq:
                self.check_delta = False
               # self.qp_scaling = self.ref_scaling
               #  print("RESUMING MAIN PROBLEM")
                self.unfeasible_cnt = "FEASIBLE"
            else:
                self.unfeasible_cnt = "RECOVERING"
                # print("NOT RESUMING MAIN PROBLEM")
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
            # state echoes
            "q": self.q.copy(),
            "dq": self.dq.copy(),
            "ddq": self.ddq.copy(),
            "trajectory_time": float(self.trajectory_time),
            "Dtrajectory_time": float(self.Dtrajectory_time),
            "DDtrajectory_time": float(self.DDtrajectory_time),
            "unfeasible_cnt": self.unfeasible_cnt,
        }

    def update_parameters(self, h):
        pass
