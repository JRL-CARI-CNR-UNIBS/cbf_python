import numpy as np
from numpy.testing import assert_allclose
import pytest
from unittest.mock import MagicMock
import quadprog

# Importazioni dal modulo fornito
from Controller.Numba_scripts.ssm_cbf_acc import (
    dmin_and_jacobian_numba,
    jacobian_psi_times_fg_fast_numba,
    compute_h_and_lie_numba,
    compute_h_and_constraints_numba
)

# Importazioni per il controller di alto livello
from Controller.optimal_cbf_task_controller import BCFOptimalController, ControllerConfig
import Controller.Numba_scripts.numba_kernels

# Tolleranze globali per le validazioni numeriche
RTOL = 1e-4
ATOL = 1e-5
EPS = 1e-6  # Passo di perturbazione per differenze finite


def test_dmin_jacobian_finite_differences():
    """
    Verifica oggettivamente che il gradiente analitico della distanza minima
    rispetto a [d, v_r, v_h, a_h] coincida con l'approssimazione numerica tramite differenze finite.
    Isola specificamente la correttezza della regola della catena all'istante t4.
    """
    d_val = 2.0
    v_r = -1.0
    v_h = 0.5
    a_h = 0.2
    tr = 0.15
    a_max = 2.5
    atol_numba = 1e-12

    # Calcolo analitico
    d_min_analitico, jac_analitico = dmin_and_jacobian_numba(
        d_val, v_r, v_h, a_h, tr, a_max, atol_numba
    )

    # Calcolo numerico (Differenze Finite Centrali)
    jac_numerico = np.zeros(4)

    # Derivata rispetto a d
    d_p, _ = dmin_and_jacobian_numba(d_val + EPS, v_r, v_h, a_h, tr, a_max, atol_numba)
    d_m, _ = dmin_and_jacobian_numba(d_val - EPS, v_r, v_h, a_h, tr, a_max, atol_numba)
    jac_numerico[0] = (d_p - d_m) / (2 * EPS)

    # Derivata rispetto a v_r
    d_p, _ = dmin_and_jacobian_numba(d_val, v_r + EPS, v_h, a_h, tr, a_max, atol_numba)
    d_m, _ = dmin_and_jacobian_numba(d_val, v_r - EPS, v_h, a_h, tr, a_max, atol_numba)
    jac_numerico[1] = (d_p - d_m) / (2 * EPS)

    # Derivata rispetto a v_h
    d_p, _ = dmin_and_jacobian_numba(d_val, v_r, v_h + EPS, a_h, tr, a_max, atol_numba)
    d_m, _ = dmin_and_jacobian_numba(d_val, v_r, v_h - EPS, a_h, tr, a_max, atol_numba)
    jac_numerico[2] = (d_p - d_m) / (2 * EPS)

    # Derivata rispetto ad a_h
    d_p, _ = dmin_and_jacobian_numba(d_val, v_r, v_h, a_h + EPS, tr, a_max, atol_numba)
    d_m, _ = dmin_and_jacobian_numba(d_val, v_r, v_h, a_h - EPS, tr, a_max, atol_numba)
    jac_numerico[3] = (d_p - d_m) / (2 * EPS)

    # Verifica
    assert_allclose(
        jac_analitico,
        jac_numerico,
        rtol=RTOL,
        atol=ATOL,
        err_msg="Il gradiente analitico di dmin diverge dalle differenze finite. Errore nella derivazione."
    )


def test_jacobian_psi_tangential_division():
    """
    Verifica che il kernel esegua correttamente la divisione per la distanza 'd'
    durante la valutazione dei moti tangenziali, conformemente all'Equazione (12).
    """
    p_r = np.array([0.0, 0.0, 0.0])
    p_h = np.array([2.0, 0.0, 0.0])  # Distanza d = 2.0, versore u = [-1, 0, 0]

    # Velocità puramente tangenziale lungo l'asse Y
    v_r = np.array([0.0, 1.0, 0.0])
    v_h = np.array([0.0, 0.0, 0.0])

    atol_numba = 1e-12

    Jpsi_f, _ = jacobian_psi_times_fg_fast_numba(p_r, p_h, v_r, v_h, atol_numba)

    # La velocità tangenziale v_r_tan è [0, 1, 0].
    # La derivata formale richiede: (v_r_tan * v_diff) / d  -> (1.0 * 1.0) / 2.0 = 0.5
    expected_Jpsi_f1 = 0.5

    assert_allclose(
        Jpsi_f[1],
        expected_Jpsi_f1,
        rtol=1e-5,
        err_msg="Errore geometrico: Jpsi_f non esegue la divisione per la distanza d per i vettori tangenziali."
    )


def test_lie_derivatives_finite_differences():
    """
    Valida oggettivamente le derivate di Lie simulando l'evoluzione dello stato cartesiano.
    """
    dt = 1e-5

    p_r = np.array([0.0, 0.0, 0.0])
    p_h = np.array([2.0, 0.5, 0.0])
    v_r = np.array([0.5, -0.2, 0.0])
    v_h = np.array([-0.3, 0.1, 0.0])
    a_h_vec = np.array([0.1, 0.0, 0.0])

    Tr, a_s, C, atol_numba = 0.15, 2.5, 0.25, 1e-12

    # Calcolo analitico
    h_base, Lie_f_h, Lie_g_h, _, _, _ = compute_h_and_lie_numba(
        p_r, p_h, v_r, v_h, Tr, a_s, C, a_h_vec, atol_numba
    )

    # Verifica L_f h (Evoluzione autonoma, acc robot = 0)
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
        err_msg="Lie_f_h diverge dalla variazione temporale autonoma."
    )

    # Verifica L_g h (Evoluzione forzata, acc robot = [1,0,0])
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
        err_msg="Lie_g_h[0] diverge dalla sensibilità numerica all'attuazione."
    )


def test_cbf_joint_space_mapping():
    """
    Verifica che il vincolo sull'accelerazione dei giunti sia la proiezione
    esatta del gradiente cartesiano tramite lo Jacobiano traslazionale.
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
        err_msg="Mappaggio errato: la riga del QP non corrisponde alla proiezione L_g_h * J."
    )

    expected_bound = - (Lie_g_h @ (dJlin @ dq)) - Lie_f_h - gamma * h
    assert_allclose(
        bound, expected_bound, rtol=1e-5,
        err_msg="Termine noto errato: mancata compensazione della dinamica autonoma."
    )


def test_qp_strict_cbf_satisfaction():
    """
    Forza uno scenario di quasi collisione e verifica che il risolutore QP
    (quadprog) produca un vettore che soddisfa strettamente la disuguaglianza CBF.
    """
    nq = 6
    n_constraints = 1

    P = np.eye(nq + 1) * 0.01
    b = np.zeros(nq + 1)
    b[0] = -10.0  # Forza l'ottimizzatore a richiedere un'accelerazione pericolosa

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
        f"Violazione vincolo QP. Ottenuto: {margin_satisfaction}, Richiesto: {bound}"
    )


def test_controller_fallback_on_infeasible_qp():
    """
    Verifica oggettivamente che il BCFOptimalController inneschi
    la modalità di fallback in presenza di un QP matematicamente infeasible,
    simulando la risposta del risolutore.
    """
    import pinocchio as pin
    import quadprog

    cfg = ControllerConfig()

    # Crea un modello Pinocchio reale a 6 gradi di libertà
    model = pin.buildSampleModelManipulator()
    nq = model.nq

    cfg.prefix = ""
    cfg.tool_frame = model.frames[-1].name
    cfg.elbow_frame = model.frames[-2].name

    wrapper = MagicMock()
    wrapper.model = model

    controller = BCFOptimalController(wrapper, cfg, useCbf=True)

    # Intercettiamo direttamente il risolutore matematico
    original_solve_qp = quadprog.solve_qp
    call_count = 0

    def fake_solve_qp(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Primo tentativo (Problema primario): fallimento forzato
            raise ValueError("constraints are inconsistent")
        else:
            # Secondo tentativo (Problema di fallback rilassato): successo
            # Quadprog restituisce (soluzione, valore_funzione, vettore_vincoli, ...)
            return np.zeros(nq + 1), 0.0, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

    quadprog.solve_qp = fake_solve_qp

    try:
        out = controller.step(
            obs_pos=np.zeros((1, 3)), obs_vel=np.zeros((1, 3)), obs_acc=np.zeros((1, 3)),
            nominal_q=np.zeros(nq), nominal_Dq=np.zeros(nq), nominal_DDq=np.zeros(nq)
        )

        assert out["unfeasible_cnt"] == "UNFEASIBLE", "Stato di infeasibilità non registrato."
        assert controller.qp_scaling == 0.0, "Lo scaling non è stato forzato a zero nel fallback."
        assert controller.check_delta is True, "Flag check_delta non attivato per la fase di recupero."

    finally:
        # Ripristino indispensabile del risolutore per non corrompere altri test
        quadprog.solve_qp = original_solve_qp