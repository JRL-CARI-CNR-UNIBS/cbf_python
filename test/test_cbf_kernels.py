"""Unit tests for low-level CBF acceleration kernels and Lie derivatives."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from cbf_python.controllers.kernels.ssm_cbf_acc import (
    dmin_and_jacobian_numba,
    jacobian_psi_times_fg_fast_numba,
    compute_h_and_lie_numba,
    compute_h_lie_and_grad_numba,
    compute_h_and_constraints_numba,
    segment_to_segment_distance,
)
from cbf_python.controllers.kernels.numba_kernels import (
    build_free_forced_one_step,
    fill_pos_rows,
    fill_tube_rows,
    fill_scaling_rows,
    assemble_qp_inplace,
)

RTOL = 1e-4
ATOL = 1e-5
EPS = 1e-6


def test_dmin_jacobian_finite_differences():
    """Verify analytical gradient of dmin coincides with central finite differences."""
    d_val, v_r, v_h, a_h = 2.0, -1.0, 0.5, 0.2
    tr, a_max, atol_numba = 0.15, 2.5, 1e-12

    d_min_anal, jac_anal = dmin_and_jacobian_numba(d_val, v_r, v_h, a_h, tr, a_max, atol_numba)

    jac_num = np.zeros(4)
    # Derivative w.r.t d
    dp, _ = dmin_and_jacobian_numba(d_val + EPS, v_r, v_h, a_h, tr, a_max, atol_numba)
    dm, _ = dmin_and_jacobian_numba(d_val - EPS, v_r, v_h, a_h, tr, a_max, atol_numba)
    jac_num[0] = (dp - dm) / (2 * EPS)

    # Derivative w.r.t v_r
    dp, _ = dmin_and_jacobian_numba(d_val, v_r + EPS, v_h, a_h, tr, a_max, atol_numba)
    dm, _ = dmin_and_jacobian_numba(d_val, v_r - EPS, v_h, a_h, tr, a_max, atol_numba)
    jac_num[1] = (dp - dm) / (2 * EPS)

    # Derivative w.r.t v_h
    dp, _ = dmin_and_jacobian_numba(d_val, v_r, v_h + EPS, a_h, tr, a_max, atol_numba)
    dm, _ = dmin_and_jacobian_numba(d_val, v_r, v_h - EPS, a_h, tr, a_max, atol_numba)
    jac_num[2] = (dp - dm) / (2 * EPS)

    # Derivative w.r.t a_h
    dp, _ = dmin_and_jacobian_numba(d_val, v_r, v_h, a_h + EPS, tr, a_max, atol_numba)
    dm, _ = dmin_and_jacobian_numba(d_val, v_r, v_h, a_h - EPS, tr, a_max, atol_numba)
    jac_num[3] = (dp - dm) / (2 * EPS)

    assert_allclose(jac_anal, jac_num, rtol=RTOL, atol=ATOL)


def test_jacobian_psi_times_fg_finite_differences():
    """Verify state-space mapping Jacobian against numerical derivatives."""
    p_r = np.array([1.2, -0.4, 0.8])
    p_h = np.array([0.5, 0.2, 0.1])
    v_r = np.array([-0.3, 0.1, -0.2])
    v_h = np.array([0.1, -0.05, 0.05])
    a_h = np.array([0.2, -0.1, 0.15])
    atol = 1e-12

    Jf_anal, Jg_anal = jacobian_psi_times_fg_fast_numba(p_r, p_h, v_r, v_h, a_h, atol)

    # Validate Jf: time derivative of psi under free dynamics
    def eval_psi(pr, ph, vr, vh, ah):
        r = pr - ph
        d = np.linalg.norm(r)
        u = r / d
        return np.array([d, np.dot(u, vr), np.dot(u, vh), np.dot(u, ah)])

    psi_p = eval_psi(p_r + EPS * v_r, p_h + EPS * v_h, v_r, v_h, a_h)
    psi_m = eval_psi(p_r - EPS * v_r, p_h - EPS * v_h, v_r, v_h, a_h)
    Jf_num = (psi_p - psi_m) / (2 * EPS)
    assert_allclose(Jf_anal, Jf_num, rtol=RTOL, atol=ATOL)

    # Validate Jg: directional derivatives w.r.t robot acceleration input
    Jg_num = np.zeros((4, 3))
    for j in range(3):
        acc = np.zeros(3)
        acc[j] = 1.0
        p_plus = eval_psi(p_r, p_h, v_r + EPS * acc, v_h, a_h)
        p_minus = eval_psi(p_r, p_h, v_r - EPS * acc, v_h, a_h)
        Jg_num[:, j] = (p_plus - p_minus) / (2 * EPS)

    assert_allclose(Jg_anal, Jg_num, rtol=RTOL, atol=ATOL)


def test_lie_derivatives_consistency():
    """Verify that compute_h_and_lie_numba correctly chains partial derivatives."""
    p_r = np.array([1.0, 0.5, 0.2])
    p_h = np.array([1.5, 0.8, 0.2])
    v_r = np.array([-0.5, -0.2, 0.0])
    v_obs = np.array([0.1, 0.0, 0.0])
    obs_acc = np.array([0.0, 0.0, 0.0])
    Tr, a_s, C, atol = 0.15, 2.5, 0.25, 1e-12

    h_val, Lf_h, Lg_h, d, vr, vh, h_chi = compute_h_and_lie_numba(
        p_r, p_h, v_r, v_obs, Tr, a_s, C, obs_acc, atol
    )
    assert d > 0.0
    assert isinstance(Lf_h, float)
    assert Lg_h.shape == (3,)
    assert isinstance(h_chi, float)


def test_constraint_assembly_vectorization():
    """Verify consistency between compute_h_and_constraints_numba and manual contraction."""
    nq = 6
    p_r = np.array([0.8, 0.1, 0.5])
    p_h = np.array([1.0, 0.2, 0.4])
    v_r = np.array([-0.1, 0.0, 0.1])
    v_obs = np.array([0.05, -0.05, 0.0])
    obs_acc = np.array([0.01, 0.0, 0.0])
    Tr, a_s, C, atol = 0.15, 2.5, 0.25, 1e-12
    gamma = 5.0
    delta_H = 0.15

    np.random.seed(42)
    Jlin = np.random.randn(3, nq)
    dJlin = np.random.randn(3, nq)
    dq = np.random.randn(nq)

    h, row, bound, d, vr, vh = compute_h_and_constraints_numba(
        p_r, p_h, v_r, v_obs, Tr, a_s, C, obs_acc, atol, Jlin, dJlin, dq, gamma, True, delta_H
    )

    _, Lf_h, Lg_h, _, _, _, h_chi = compute_h_and_lie_numba(
        p_r, p_h, v_r, v_obs, Tr, a_s, C, obs_acc, atol
    )

    row_expected = Lg_h @ Jlin
    bound_expected = -(Lg_h @ (dJlin @ dq)) - Lf_h - gamma * h - h_chi * delta_H

    assert_allclose(row, row_expected, rtol=RTOL, atol=ATOL)
    assert_allclose(bound, bound_expected, rtol=RTOL, atol=ATOL)


def test_numba_qp_assembly():
    """Verify inplace QP assembly kernels."""
    nq = 6
    P2 = np.zeros((nq + 1, nq + 1))
    b_pos = np.zeros(nq + 1)
    b_vel = np.zeros(nq + 1)
    b_scaling = np.zeros(nq + 1)
    q = np.zeros(nq)
    dq = np.zeros(nq)
    nom_q = np.zeros(nq)
    nom_dq = np.zeros(nq)

    from cbf_python.controllers.kernels.numba_kernels import assemble_objective_parts_inplace
    assemble_objective_parts_inplace(P2, b_pos, b_vel, b_scaling, q, dq, nom_q, nom_dq, 1.0, 0.002, 1.0)

    assert P2.shape == (nq + 1, nq + 1)
    assert b_pos.shape == (nq + 1,)


def test_fill_pos_rows():
    """Verify fill_pos_rows linear inequalities match q_min <= q(k+1) <= q_max."""
    nq = 6
    Ts = 0.002
    FreePos, ForcedPos, FreeVel, ForcedVel = build_free_forced_one_step(Ts, nq)

    q = np.array([0.5, -1.0, 1.2, 0.0, -0.5, 0.8])
    dq = np.array([0.1, -0.2, 0.05, -0.1, 0.3, -0.05])
    x0 = np.concatenate([q, dq])

    q_min = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
    q_max = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    A = np.zeros((2 * nq, nq))
    c = np.zeros(2 * nq)
    row = fill_pos_rows(A, c, 0, nq, FreePos, ForcedPos, x0, q_min, q_max)
    assert row == 2 * nq

    # Test random acceleration inputs
    for _ in range(50):
        ddq = np.random.randn(nq) * 50.0
        q_next = FreePos @ x0 + ForcedPos @ ddq
        # A @ ddq >= c should be elementwise equivalent to q_min <= q_next <= q_max
        ineq_satisfaction = (A @ ddq >= c - 1e-12)
        limits_satisfaction_upper = (q_next <= q_max + 1e-12)
        limits_satisfaction_lower = (q_next >= q_min - 1e-12)

        assert np.all(ineq_satisfaction[:nq] == limits_satisfaction_upper)
        assert np.all(ineq_satisfaction[nq:] == limits_satisfaction_lower)


def test_segment_to_segment_distance():
    """Verify Ericson segment-to-segment distance algorithm across parallel, skew, and point cases."""
    # 1. Parallel segments with radii
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    q1 = np.array([0.0, 1.0, 0.0])
    q2 = np.array([1.0, 1.0, 0.0])
    r1, r2 = 0.1, 0.1
    d_surf, c1, c2, pr, ph, s, t = segment_to_segment_distance(p1, p2, q1, q2, r1, r2)
    assert abs(d_surf - 0.8) < 1e-10
    assert abs(s - 0.0) < 1e-10
    assert abs(t - 0.0) < 1e-10
    assert_allclose(c1, np.array([0.0, 0.0, 0.0]), atol=1e-10)
    assert_allclose(c2, np.array([0.0, 1.0, 0.0]), atol=1e-10)

    # 2. Orthogonal skew segments crossing at midpoints
    p1 = np.array([0.0, 0.5, 0.0])
    p2 = np.array([1.0, 0.5, 0.0])
    q1 = np.array([0.5, 0.0, 1.0])
    q2 = np.array([0.5, 1.0, 1.0])
    d_surf, c1, c2, pr, ph, s, t = segment_to_segment_distance(p1, p2, q1, q2, 0.0, 0.0)
    assert abs(d_surf - 1.0) < 1e-10
    assert abs(s - 0.5) < 1e-10
    assert abs(t - 0.5) < 1e-10
    assert_allclose(c1, np.array([0.5, 0.5, 0.0]), atol=1e-10)
    assert_allclose(c2, np.array([0.5, 0.5, 1.0]), atol=1e-10)

    # 3. Point to line segment
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    q_pt = np.array([0.5, 0.5, 0.0])
    d_surf, c1, c2, pr, ph, s, t = segment_to_segment_distance(p1, p2, q_pt, q_pt, 0.05, 0.05)
    assert abs(d_surf - 0.4) < 1e-10
    assert abs(s - 0.5) < 1e-10
    assert abs(t - 0.0) < 1e-10


def test_robust_covariance_constraint_assembly():
    """Verify that robust covariance term correctly scales with feature gradient and beta."""
    nq = 6
    p_r = np.array([0.8, 0.1, 0.5])
    p_h = np.array([1.0, 0.2, 0.4])
    v_r = np.array([-0.2, 0.0, 0.1])
    v_obs = np.array([0.1, -0.05, 0.0])
    obs_acc = np.array([0.05, 0.0, 0.0])
    Tr, a_s, C, atol = 0.15, 2.5, 0.25, 1e-12
    gamma = 5.0
    tc = 0.002

    np.random.seed(123)
    Jlin = np.random.randn(3, nq)
    dJlin = np.random.randn(3, nq)
    dq = np.random.randn(nq)

    sigma_psi = np.array([0.015, 0.0, 0.072, 0.196])
    beta = 3.291

    h, row, bound_cov, d, vr, vh = compute_h_and_constraints_numba(
        p_r, p_h, v_r, v_obs, Tr, a_s, C, obs_acc, atol, Jlin, dJlin, dq, gamma, True,
        delta_H=0.0, tc=tc, sigma_psi=sigma_psi, beta=beta,
    )

    h_zero, row_zero, bound_zero, _, _, _ = compute_h_and_constraints_numba(
        p_r, p_h, v_r, v_obs, Tr, a_s, C, obs_acc, atol, Jlin, dJlin, dq, gamma, True,
        delta_H=0.0, tc=tc, sigma_psi=np.zeros(4), beta=beta,
    )

    # Adding uncertainty must tighten the constraint (increase the lower bound)
    assert bound_cov > bound_zero
    # Difference must equal gamma * beta * sigma_eff
    _, _, _, _, _, _, _, Jh_psi = compute_h_lie_and_grad_numba(p_r, p_h, v_r, v_obs, Tr, a_s, C, obs_acc, atol, tc)
    sigma_eff = np.sqrt(
        (Jh_psi[0] * sigma_psi[0]) ** 2 +
        (Jh_psi[1] * sigma_psi[1]) ** 2 +
        (Jh_psi[2] * sigma_psi[2]) ** 2 +
        (Jh_psi[3] * sigma_psi[3]) ** 2
    )
    expected_diff = gamma * (beta * sigma_eff)
    assert_allclose(bound_cov - bound_zero, expected_diff, rtol=RTOL, atol=ATOL)


def test_relative_degree_restoration_at_causality_lower_bound():
    """Verify that setting the evaluation horizon lower bound to tc = dt restores relative degree 1."""
    d_val = 1.0
    v_r = 0.5   # Robot moving away
    v_h = -0.5  # Human moving away
    a_h = 0.0
    tr = 0.15
    a_max = 2.5
    atol = 1e-12

    # Case A: Legacy / Paper Formulation (tc = 0)
    # Minimizer is at tau* = 0, causing d(d_min)/d(v_r) = 0 and vanishing control Lie derivative
    d_min_0, jac_0 = dmin_and_jacobian_numba(d_val, v_r, v_h, a_h, tr, a_max, atol, tc=0.0)
    assert_allclose(d_min_0, 1.0)
    assert_allclose(jac_0[1], 0.0)  # Relative degree loss: zero sensitivity to velocity!

    # Case B: Implemented Formulation (tc = dt = 0.002 s)
    # Evaluation horizon is tau in [tc, Tf], ensuring tau* = tc > 0
    tc = 0.002
    d_min_tc, jac_tc = dmin_and_jacobian_numba(d_val, v_r, v_h, a_h, tr, a_max, atol, tc=tc)
    assert d_min_tc > d_val
    assert_allclose(jac_tc[1], tc)   # Uniform relative degree 1 restored: d(d_min)/d(v_r) = tc!
    assert_allclose(jac_tc[2], -tc)
    assert_allclose(jac_tc[3], -0.5 * (tc ** 2))

    # Verify control Lie derivative L_g h is strictly non-zero
    p_r = np.array([1.0, 0.0, 0.0])
    p_h = np.array([0.0, 0.0, 0.0])
    v_r_vec = np.array([0.5, 0.0, 0.0])
    v_h_vec = np.array([-0.5, 0.0, 0.0])
    a_h_vec = np.array([0.0, 0.0, 0.0])
    C = 0.25

    _, Lf_0, Lg_0, _, _, _, _, _ = compute_h_lie_and_grad_numba(p_r, p_h, v_r_vec, v_h_vec, tr, a_max, C, a_h_vec, atol, tc=0.0)
    _, Lf_tc, Lg_tc, _, _, _, _, _ = compute_h_lie_and_grad_numba(p_r, p_h, v_r_vec, v_h_vec, tr, a_max, C, a_h_vec, atol, tc=tc)

    assert_allclose(np.linalg.norm(Lg_0), 0.0)
    assert_allclose(np.linalg.norm(Lg_tc), tc)  # Exactly equals dt along unit line of sight!



