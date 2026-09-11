"""Unit tests for CBF controllers (Optimal, Polynomial, Gaussian, PID)."""

import numpy as np
import pytest
from sharework import loadSharework

from cbf_python.controllers.base_optimal_controller import BCFOptimalController, ControllerConfig
from cbf_python.controllers.polynomial_controller import PolynomialControllerConfig, PolynomialOptimalController
from cbf_python.controllers.gaussian_controller import GaussianControllerConfig, GaussianController, GaussianSet
from cbf_python.controllers.pid_cbf_controller import UR10CBFController
from cbf_python.utils.simulation_helpers import HOME, UR10E_JOINTS


@pytest.fixture(scope="module")
def model_wrapper():
    return loadSharework(UR10E_JOINTS)


def test_optimal_controller_step(model_wrapper):
    """Verify BCFOptimalController step and feasible QP solution."""
    cfg = ControllerConfig(Tc=0.002)
    cfg.lambda_pos = 2000.0
    cfg.lambda_vel = 1.0
    cfg.lambda_scaling = 20.0
    cfg.lambda_acc = 1e-10
    cfg.gamma = 5.0
    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True)
    ctrl.reset_state(HOME)

    obs_pos = np.array([[1.5, 0.0, 0.5]])
    obs_vel = np.array([[0.0, 0.0, 0.0]])
    obs_acc = np.array([[0.0, 0.0, 0.0]])

    nominal_q = HOME.copy()
    nominal_Dq = np.zeros(6)
    nominal_DDq = np.zeros(6)

    out = ctrl.step(
        obs_pos=obs_pos,
        obs_vel=obs_vel,
        obs_acc=obs_acc,
        nominal_q=nominal_q,
        nominal_Dq=nominal_Dq,
        nominal_DDq=nominal_DDq,
    )

    assert "q" in out
    assert "dq" in out
    assert "ddq" in out
    assert "Dtrajectory_time" in out
    assert out["q"].shape == (6,)
    assert out["unfeasible"] == "FEASIBLE"


def test_polynomial_controller_step(model_wrapper):
    """Verify PolynomialOptimalController updates weights dynamically."""
    cfg = PolynomialControllerConfig(Tc=0.002)
    cfg.lambda_0_pos = 2000.0
    cfg.lambda_0_vel = 1.0
    cfg.lambda_0_acc = 1e-10
    cfg.lambda_0_scaling = 20.0
    cfg.gamma_0 = 5.0

    cfg.lambda_f_pos = 1000.0
    cfg.lambda_f_vel = 0.5
    cfg.lambda_f_acc = 1e-10
    cfg.lambda_f_scaling = 10.0
    cfg.gamma_f = 2.0

    cfg.generate_poly_dict()
    ctrl = PolynomialOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True)
    ctrl.reset_state(HOME)

    obs_pos = np.array([[1.5, 0.0, 0.5]])
    obs_vel = np.array([[0.0, 0.0, 0.0]])
    obs_acc = np.array([[0.0, 0.0, 0.0]])

    out = ctrl.step(
        obs_pos=obs_pos,
        obs_vel=obs_vel,
        obs_acc=obs_acc,
        nominal_q=HOME,
        nominal_Dq=np.zeros(6),
        nominal_DDq=np.zeros(6),
    )

    assert "q" in out
    assert out["Dtrajectory_time"] >= 0.0


def test_gaussian_controller_step(model_wrapper):
    """Verify GaussianController with GaussianSet."""
    cfg = GaussianControllerConfig(Tc=0.002)
    cfg.lambda_pos = 2000.0
    cfg.lambda_vel = 1.0
    cfg.lambda_scaling = 20.0
    cfg.lambda_acc = 1e-10
    cfg.gamma = 5.0

    gs = GaussianSet(
        means={"h": 0.0, "d": 0.5, "v_rel": 0.0},
        covariance=np.eye(3) * 0.01,
        lambda_ref={"pos": 1000.0, "vel": 1.0, "acc": 1e-10, "scaling": 50.0, "gamma": 5.0},
    )
    cfg.gaussian_sets.append(gs)
    cfg.n_gaussian_sets = len(cfg.gaussian_sets)
    cfg.precompute_gaussian_parameters()

    ctrl = GaussianController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True)
    ctrl.reset_state(HOME)

    out = ctrl.step(
        obs_pos=np.array([[1.5, 0.0, 0.5]]),
        obs_vel=np.array([[0.0, 0.0, 0.0]]),
        obs_acc=np.array([[0.0, 0.0, 0.0]]),
        nominal_q=HOME,
        nominal_Dq=np.zeros(6),
        nominal_DDq=np.zeros(6),
    )

    assert "q" in out


def test_pid_controller_step(model_wrapper):
    """Verify UR10CBFController Cartesian PID + CBF."""
    import pinocchio as pin

    model = model_wrapper.model
    ctrl = UR10CBFController(
        model=model,
        tool_frame_name="ur10e_wrist_3_joint",
        useCbf=True,
    )
    ctrl.reset_state(HOME)

    data = model.createData()
    pin.forwardKinematics(model, data, HOME)
    pin.updateFramePlacements(model, data)
    goal_pose = data.oMf[model.getFrameId("ur10e_wrist_3_joint")]
    twist_goal = np.zeros(6)

    out = ctrl.step(
        goal_pose=goal_pose,
        twist_goal=twist_goal,
        obstacle_positions=np.array([[2.0, 0.0, 0.5]]),
        obstacle_velocities=np.array([[0.0, 0.0, 0.0]]),
        obstacle_accelerations=np.array([[0.0, 0.0, 0.0]]),
    )

    assert "q" in out
    assert "trajectory_error" in out
    assert out["trajectory_error"] < 0.05


def test_controller_respects_joint_position_limits(model_wrapper):
    """Verify that optimal and PID controllers respect joint position limits."""
    cfg = ControllerConfig(Tc=0.002)
    cfg.lambda_pos = 2000.0
    cfg.lambda_vel = 1.0
    cfg.lambda_scaling = 20.0
    cfg.lambda_acc = 1e-10
    cfg.gamma = 5.0
    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=False)

    # Set custom tight joint limits for testing
    ctrl.q_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ctrl.q_min = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

    # Initial state close to upper limit with positive velocity
    ctrl.reset_state(np.array([0.99, 0.99, 0.99, 0.99, 0.99, 0.99]))
    ctrl.dq = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Nominal trajectory requests driving far beyond upper limit
    nominal_q = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    nominal_Dq = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    nominal_DDq = np.zeros(6)

    out = ctrl.step(
        obs_pos=np.empty((0, 3)),
        obs_vel=np.empty((0, 3)),
        obs_acc=np.empty((0, 3)),
        nominal_q=nominal_q,
        nominal_Dq=nominal_Dq,
        nominal_DDq=nominal_DDq,
    )

    # Robot next position must strictly respect upper limit
    assert np.all(out["q"] <= ctrl.q_max + 1e-8)
    assert np.all(out["q"] >= ctrl.q_min - 1e-8)


def test_infeasible_fallback_deterministic_braking(model_wrapper):
    """Verify deterministic braking fallback (Eq. 16) when QP is infeasible."""
    cfg = ControllerConfig(Tc=0.002)
    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=False)

    # Force an impossible QP constraint: q_min > q_max
    ctrl.q_min = np.array([2.0] * 6)
    ctrl.q_max = np.array([-2.0] * 6)
    ctrl.reset_state(HOME)
    ctrl.dq = np.array([1.0, -2.0, 0.5, -0.1, 3.0, 0.0])
    dq_init = ctrl.dq.copy()

    out = ctrl.step(
        obs_pos=np.empty((0, 3)),
        obs_vel=np.empty((0, 3)),
        obs_acc=np.empty((0, 3)),
        nominal_q=HOME,
        nominal_Dq=np.zeros(6),
        nominal_DDq=np.zeros(6),
    )

    assert out["unfeasible"] == "UNFEASIBLE"
    expected_ddq = np.clip(-dq_init / cfg.Tc, -cfg.DDq_max, cfg.DDq_max)
    assert np.allclose(out["ddq"], expected_ddq)


def test_pid_infeasible_fallback_deterministic_braking(model_wrapper):
    """Verify PID controller deterministic braking fallback (Eq. 16)."""
    import pinocchio as pin

    model = model_wrapper.model
    ctrl = UR10CBFController(
        model=model,
        tool_frame_name="ur10e_wrist_3_joint",
        useCbf=True,
    )
    ctrl.reset_state(HOME)
    ctrl.dq = np.array([1.5, -2.0, 0.5, -0.8, 1.0, -0.5])
    dq_init = ctrl.dq.copy()

    # Contradictory joint limits to make QP infeasible
    ctrl.q_min = np.array([2.0] * 6)
    ctrl.q_max = np.array([-2.0] * 6)

    data = model.createData()
    pin.forwardKinematics(model, data, HOME)
    pin.updateFramePlacements(model, data)
    goal_pose = data.oMf[model.getFrameId("ur10e_wrist_3_joint")]

    out = ctrl.step(
        goal_pose=goal_pose,
        twist_goal=np.zeros(6),
        obstacle_positions=np.array([[2.0, 0.0, 0.5]]),
        obstacle_velocities=np.array([[0.0, 0.0, 0.0]]),
        obstacle_accelerations=np.array([[0.0, 0.0, 0.0]]),
    )

    expected_ddq = np.clip(-dq_init / ctrl.Tc, -ctrl.DDq_max, ctrl.DDq_max)
    assert np.allclose(out["ddq"], expected_ddq)


def test_monotonic_tube_recovery(model_wrapper):
    """Verify monotonic tube shrinking and progression suspension (Eq. 20-21)."""
    cfg = ControllerConfig(Tc=0.002)
    cfg.delta_q_max = np.array([0.05] * 6)
    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=False)
    ctrl.reset_state(HOME)

    # Trigger recovery by simulating a disturbance that displaced the robot outside the tube
    ctrl.check_delta = True
    ctrl.delta_q_max = np.array([0.20] * 6)
    ctrl.Dtrajectory_time = 0.8

    # Step 1: tracking error is 0.15 rad. delta_temp should shrink to 0.15
    nominal_q = HOME + 0.15
    out = ctrl.step(
        obs_pos=np.empty((0, 3)),
        obs_vel=np.empty((0, 3)),
        obs_acc=np.empty((0, 3)),
        nominal_q=nominal_q,
        nominal_Dq=np.zeros(6),
        nominal_DDq=np.zeros(6),
    )
    assert ctrl.check_delta is True
    assert out["unfeasible"] == "RECOVERING"
    # Progression suspended: Dtrajectory_time is held at 0.0
    assert ctrl.Dtrajectory_time <= 0.05
    # Tube has shrunk monotonically
    assert np.all(ctrl.delta_q_max <= 0.20 + 1e-6)

    # Step 2: robot moves back inside nominal tube (nominal_q = HOME)
    ctrl.reset_state(HOME)
    out2 = ctrl.step(
        obs_pos=np.empty((0, 3)),
        obs_vel=np.empty((0, 3)),
        obs_acc=np.empty((0, 3)),
        nominal_q=HOME,
        nominal_Dq=np.zeros(6),
        nominal_DDq=np.zeros(6),
    )
    # Recovery terminates
    assert ctrl.check_delta is False
    assert out2["unfeasible"] == "FEASIBLE"
    assert np.allclose(ctrl.delta_q_max, cfg.delta_q_max)


def test_deadline_missed_cycles(model_wrapper):
    """Verify zero-order hold (N_miss=1) and emergency braking (N_miss>1) (Eq. 17-19)."""
    cfg = ControllerConfig(Tc=0.002)
    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=False)
    ctrl.reset_state(HOME)

    # Step 1: Normal step
    out1 = ctrl.step(
        obs_pos=np.empty((0, 3)),
        obs_vel=np.empty((0, 3)),
        obs_acc=np.empty((0, 3)),
        nominal_q=HOME,
        nominal_Dq=np.zeros(6),
        nominal_DDq=np.zeros(6),
        deadline_missed=False,
    )
    u_prev = ctrl.u_prev.copy()
    assert ctrl.N_miss == 0

    # Step 2: N_miss = 1 -> Zero order hold (u_k = u_{k-1})
    out2 = ctrl.step(
        obs_pos=np.empty((0, 3)),
        obs_vel=np.empty((0, 3)),
        obs_acc=np.empty((0, 3)),
        nominal_q=HOME,
        nominal_Dq=np.zeros(6),
        nominal_DDq=np.zeros(6),
        deadline_missed=True,
    )
    assert ctrl.N_miss == 1
    assert np.allclose(out2["ddq"], u_prev[:-1])

    # Step 3: N_miss = 2 -> Emergency braking u^bk
    dq_before = ctrl.dq.copy()
    out3 = ctrl.step(
        obs_pos=np.empty((0, 3)),
        obs_vel=np.empty((0, 3)),
        obs_acc=np.empty((0, 3)),
        nominal_q=HOME,
        nominal_Dq=np.zeros(6),
        nominal_DDq=np.zeros(6),
        deadline_missed=True,
    )
    assert ctrl.N_miss == 2
    expected_braking = np.clip(-dq_before / cfg.Tc, -cfg.DDq_max, cfg.DDq_max)
    assert np.allclose(out3["ddq"], expected_braking)

    # Step 4: Solved in time -> N_miss resets to 0
    out4 = ctrl.step(
        obs_pos=np.empty((0, 3)),
        obs_vel=np.empty((0, 3)),
        obs_acc=np.empty((0, 3)),
        nominal_q=HOME,
        nominal_Dq=np.zeros(6),
        nominal_DDq=np.zeros(6),
        deadline_missed=False,
    )
    assert ctrl.N_miss == 0

