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
