import numpy as np
from numpy.testing import assert_allclose
import pytest
from unittest.mock import MagicMock
import quadprog

# Import kernels from SSM-CBF Numba module
from Controller.Numba_scripts.ssm_cbf_acc import (
    dmin_and_jacobian_numba,
    jacobian_psi_times_fg_fast_numba,
    compute_h_and_lie_numba,
    compute_h_and_constraints_numba,
)

# Import high-level optimal controller
from Controller.optimal_cbf_task_controller import BCFOptimalController, ControllerConfig
import Controller.Numba_scripts.numba_kernels

# Global numerical validation tolerances
RTOL = 1e-4
ATOL = 1e-5
EPS = 1e-6  # Perturbation step for finite differences


def test_dmin_jacobian_finite_differences():
    """
    Verifies that the analytical gradient of the minimum distance with respect to
    [d, v_r, v_h, a_h] matches central finite difference numerical approximations.
    Specifically validates the correctness of the chain rule at critical instant t4.
    """
    d_val = 2.0
    v_r = -1.0
    v_h = 0.5
    a_h = 0.2
    tr = 0.15
    a_max = 2.5
    atol_numba = 1e-12

    # Analytical gradient computation
    d_min_analitico, jac_analitico = dmin_and_jacobian_numba(
        d_val, v_r, v_h, a_h, tr, a_max, atol_numba
    )

    # Numerical gradient computation (Central Finite Differences)
    jac_numerico = np.zeros(4)

    # Derivative w.r.t. d
    d_p, _ = dmin_and_jacobian_numba(d_val + EPS, v_r, v_h, a_h, tr, a_max, atol_numba)
    d_m, _ = dmin_and_jacobian_numba(d_val - EPS, v_r, v_h, a_h, tr, a_max, atol_numba)
    jac_numerico[0] = (d_p - d_m) / (2 * EPS)

    # Derivative w.r.t. v_r
    d_p, _ = dmin_and_jacobian_numba(d_val, v_r + EPS, v_h, a_h, tr, a_max, atol_numba)
    d_m, _ = dmin_and_jacobian_numba(d_val, v_r - EPS, v_h, a_h, tr, a_max, atol_numba)
    jac_numerico[1] = (d_p - d_m) / (2 * EPS)

    # Derivative w.r.t. v_h
    d_p, _ = dmin_and_jacobian_numba(d_val, v_r, v_h + EPS, a_h, tr, a_max, atol_numba)
    d_m, _ = dmin_and_jacobian_numba(d_val, v_r, v_h - EPS, a_h, tr, a_max, atol_numba)
    jac_numerico[2] = (d_p - d_m) / (2 * EPS)

    # Derivative w.r.t. a_h
    d_p, _ = dmin_and_jacobian_numba(d_val, v_r, v_h, a_h + EPS, tr, a_max, atol_numba)
    d_m, _ = dmin_and_jacobian_numba(d_val, v_r, v_h, a_h - EPS, tr, a_max, atol_numba)
    jac_numerico[3] = (d_p - d_m) / (2 * EPS)

    # Verification assertion
    assert_allclose(
        jac_analitico,
        jac_numerico,
        rtol=RTOL,
        atol=ATOL,
        err_msg="Analytical gradient of dmin diverges from finite differences.",
    )


def test_jacobian_psi_tangential_division():
    """
    Verifies that the kernel correctly evaluates division by distance 'd'
    during tangential motion evaluation according to the measurement Jacobian.
    """
    p_r = np.array([0.0, 0.0, 0.0])
    p_h = np.array([2.0, 0.0, 0.0])  # Distance d = 2.0, unit direction u = [-1, 0, 0]

    # Purely tangential velocity along Y axis
    v_r = np.array([0.0, 1.0, 0.0])
    v_h = np.array([0.0, 0.0, 0.0])

    atol_numba = 1e-12

    Jpsi_f, _ = jacobian_psi_times_fg_fast_numba(p_r, p_h, v_r, v_h, atol_numba)

    # Tangential velocity v_r_tan is [0, 1, 0].
    # Formal derivative: (v_r_tan * v_diff) / d -> (1.0 * 1.0) / 2.0 = 0.5
    expected_Jpsi_f1 = 0.5

    assert_allclose(
        Jpsi_f[1],
        expected_Jpsi_f1,
        rtol=1e-5,
        err_msg="Geometric error: Jpsi_f does not execute division by distance d for tangential vectors.",
    )


def test_lie_derivatives_finite_differences():
    """
    Validates Lie derivatives (L_f h and L_g h) by simulating Cartesian state evolution.
    """
    dt = 1e-5

    p_r = np.array([0.0, 0.0, 0.0])
    p_h = np.array([2.0, 0.5, 0.0])
    v_r = np.array([0.5, -0.2, 0.0])
    v_h = np.array([-0.3, 0.1, 0.0])
    a_h_vec = np.array([0.1, 0.0, 0.0])

    Tr, a_s, C, atol_numba = 0.15, 2.5, 0.25, 1e-12

    # Analytical computation
    h_base, Lie_f_h, Lie_g_h, _, _, _ = compute_h_and_lie_numba(
        p_r, p_h, v_r, v_h, Tr, a_s, C, a_h_vec, atol_numba
    )

    # Verify L_f h (Autonomous drift evolution, robot acceleration = 0)
    p_r_next = p_r + v_r * dt
    p_h_next = p_h + v_h * dt + 0.5 * a_h_vec * (dt ** 2)
    v_h_next = v_h + a_h_vec * dt

    h_next, _, _, _, _, _ = compute_h_and_lie_numba(
        p_r_next, p_h_next, v_r, v_h_next, Tr, a_s, C, a_h_vec, atol_numba
    )

    Lie_f_h_numerico = (h_next - h_base) / dt

    assert_allclose(
        Lie_f_h,
        Lie_f_h_numerico,
        rtol=1e-2,
        atol=1e-3,
        err_msg="Lie_f_h diverges from numerical autonomous time variation.",
    )

    # Verify L_g h (Forced input evolution, robot acceleration = [1, 0, 0])
    acc_r = np.array([1.0, 0.0, 0.0])
    v_r_forced = v_r + acc_r * dt

    h_forced, _, _, _, _, _ = compute_h_and_lie_numba(
        p_r_next, p_h_next, v_r_forced, v_h_next, Tr, a_s, C, a_h_vec, atol_numba
    )

    Lie_g_h_numerico_x = ((h_forced - h_base) / dt) - Lie_f_h_numerico

    assert_allclose(
        Lie_g_h[0],
        Lie_g_h_numerico_x,
        rtol=1e-2,
        atol=1e-3,
        err_msg="Lie_g_h[0] diverges from numerical actuation sensitivity.",
    )


def test_cbf_joint_space_mapping():
    """
    Verifies that the joint acceleration constraint is the exact projection
    of the Cartesian barrier gradient via the translational Jacobian.
    """
    n_joints = 6
    translation_bt = np.array([0.5, 0.5, 0.5])
    obs_pos = np.array([0.5, 0.8, 0.5])
    vel_lineare = np.array([0.0, 0.1, 0.0])
    v_obs = np.array([0.0, -0.1, 0.0])
    obs_acc = np.zeros(3)
    dq = np.ones(n_joints) * 0.1

    Jlin = np.random.rand(3, n_joints)
    dJlin = np.random.rand(3, n_joints)

    Tr, a_s, C, gamma, atol = 0.15, 2.5, 0.25, 5.0, 1e-12
    HAS_CBF = True

    h, row_vec, bound, _, _, _ = compute_h_and_constraints_numba(
        translation_bt, obs_pos, vel_lineare, v_obs,
        Tr, a_s, C, obs_acc, atol, Jlin, dJlin, dq, gamma, HAS_CBF
    )

    _, Lie_f_h, Lie_g_h, _, _, _ = compute_h_and_lie_numba(
        translation_bt, obs_pos, vel_lineare, v_obs, Tr, a_s, C, obs_acc, atol
    )

    expected_row = Lie_g_h @ Jlin
    assert_allclose(
        row_vec, expected_row, rtol=1e-5,
        err_msg="Incorrect mapping: QP row does not match L_g_h * J projection.",
    )

    expected_bound = - (Lie_g_h @ (dJlin @ dq)) - Lie_f_h - gamma * h
    assert_allclose(
        bound, expected_bound, rtol=1e-5,
        err_msg="Incorrect bound term: missing autonomous drift dynamic compensation.",
    )


def test_qp_strict_cbf_satisfaction():
    """
    Forces a near-collision scenario and verifies that the QP solver
    (quadprog) returns a control decision that strictly satisfies the CBF inequality.
    """
    nq = 6
    n_constraints = 1

    P = np.eye(nq + 1) * 0.01
    b = np.zeros(nq + 1)
    b[0] = -10.0  # Force optimizer towards dangerous acceleration direction

    A = np.zeros((n_constraints, nq + 1))
    c = np.zeros(n_constraints)

    h_val = 0.05
    Lie_f_h = -2.0
    row_vec = np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    bound = -Lie_f_h - 5.0 * h_val

    A[0, :nq] = row_vec
    c[0] = bound

    u, _, _, _, _, _ = quadprog.solve_qp(P, b, A.T, c, 0)
    ddq_opt = u[:nq]

    margin_satisfaction = row_vec @ ddq_opt

    assert margin_satisfaction >= bound - 1e-8, (
        f"QP constraint violation. Obtained: {margin_satisfaction}, Required bound: {bound}"
    )


def test_controller_fallback_on_infeasible_qp():
    """
    Verifies that BCFOptimalController triggers the fallback emergency mode
    when the primary QP is infeasible, simulating solver response.
    """
    import pinocchio as pin
    import quadprog

    cfg = ControllerConfig()

    # Create a 6-DOF Pinocchio manipulator model
    model = pin.buildSampleModelManipulator()
    nq = model.nq

    cfg.prefix = ""
    cfg.tool_frame = model.frames[-1].name
    cfg.elbow_frame = model.frames[-2].name

    wrapper = MagicMock()
    wrapper.model = model

    controller = BCFOptimalController(wrapper, cfg, useCbf=True)

    # Intercept mathematical solver directly
    original_solve_qp = quadprog.solve_qp
    call_count = 0

    def fake_solve_qp(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Primary problem attempt: forced inconsistency
            raise ValueError("constraints are inconsistent")
        else:
            # Secondary fallback problem attempt: success
            return np.zeros(nq + 1), 0.0, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

    quadprog.solve_qp = fake_solve_qp

    try:
        out = controller.step(
            obs_pos=np.zeros((1, 3)), obs_vel=np.zeros((1, 3)), obs_acc=np.zeros((1, 3)),
            nominal_q=np.zeros(nq), nominal_Dq=np.zeros(nq), nominal_DDq=np.zeros(nq)
        )

        assert out["unfeasible_cnt"] == "UNFEASIBLE", "Infeasibility state not recorded."
        assert controller.qp_scaling == 0.0, "Scaling factor was not zeroed out during fallback."
        assert controller.check_delta is True, "check_delta flag was not set for tube recovery."

    finally:
        # Restore solver to avoid side effects on subsequent tests
        quadprog.solve_qp = original_solve_qp