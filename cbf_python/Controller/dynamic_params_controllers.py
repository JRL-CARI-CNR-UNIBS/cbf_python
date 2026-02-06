from Controller.optimal_cbf_task_controller import BCFOptimalController, ControllerConfig
from dataclasses import dataclass, field
import numpy as np
def compute_generic_lambda(h, ht, lambda_0, lambda_f, n, m, w):
    if h < 0.0:
        return lambda_0
    elif h >= ht:
        return lambda_f
    else:
        return lambda_0 + (lambda_f - lambda_0) * (w * (h / ht) ** n + (1 - w) * (h / ht) ** m)

@dataclass
class PolynomialControllerConfig(ControllerConfig):
    lambda_0_pos: float = 0.0
    lambda_0_vel: float = 0.0
    lambda_0_acc: float = 0.0
    lambda_0_scaling: float = 0.0
    gamma_0: float = 0.0
    delta_0: float = 0.0

    lambda_f_pos : float = 0.0
    lambda_f_vel : float = 0.0
    lambda_f_acc : float= 0.0
    lambda_f_scaling : float = 0.0
    gamma_f : float = 0.0
    delta_f : float = 0.0

    n_pos : float = 0.0
    n_vel : float = 0.0
    n_acc : float= 0.0
    n_scaling : float = 0.0
    n_gamma : float = 0.0
    n_delta : float = 0.0

    m_pos : float = 0.0
    m_vel : float = 0.0
    m_acc : float= 0.0
    m_scaling : float = 0.0
    m_gamma : float = 0.0
    m_delta : float = 0.0

    w_pos : float = 0.0
    w_vel : float = 0.0
    w_acc : float= 0.0
    w_scaling : float = 0.0
    w_gamma : float = 0.0
    w_delta : float = 0.0

    h_t : float = 0.0
    categories = ["pos", "vel", "acc", "scaling", "gamma", "delta"]



class PolynomialOptimalController(BCFOptimalController):

    def __init__(self, model_wrapper, cfg: PolynomialControllerConfig, useCbf):
        super().__init__(model_wrapper, cfg, useCbf)
        self.cfg = cfg
        

    def update_parameters(self, h):

        self.cfg.lambda_pos = compute_generic_lambda(h, self.cfg.h_t, self.cfg.lambda_0_pos, 
                                                     self.cfg.lambda_f_pos, self.cfg.n_pos,
                                                     self.cfg.m_pos, self.cfg.w_pos)
        self.cfg.lambda_vel = compute_generic_lambda(h, self.cfg.h_t, self.cfg.lambda_0_vel,
                                                     self.cfg.lambda_f_vel, self.cfg.n_vel,
                                                     self.cfg.m_vel, self.cfg.w_vel)
        self.cfg.lambda_acc = compute_generic_lambda(h, self.cfg.h_t, self.cfg.lambda_0_acc,
                                                     self.cfg.lambda_f_acc, self.cfg.n_acc,
                                                     self.cfg.m_acc, self.cfg.w_acc)
        self.cfg.lambda_scaling = compute_generic_lambda(h, self.cfg.h_t, self.cfg.lambda_0_scaling,
                                                     self.cfg.lambda_f_scaling, self.cfg.n_scaling,
                                                     self.cfg.m_scaling, self.cfg.w_scaling)
        self.cfg.gamma = compute_generic_lambda(h, self.cfg.h_t, self.cfg.gamma_0,
                                                     self.cfg.gamma_f, self.cfg.n_gamma,
                                                     self.cfg.m_gamma, self.cfg.w_gamma)
        new_delta = compute_generic_lambda(h, self.cfg.h_t, self.cfg.delta_0,
                                                     self.cfg.delta_f, self.cfg.n_delta,
                                                     self.cfg.m_delta, self.cfg.w_delta)
        self.cfg.delta_q_max[0:2] = np.deg2rad(np.array([1,1], dtype=np.float64) * new_delta)
        self.cfg.delta_q_max[2:4] = np.deg2rad(np.array([1,1], dtype=np.float64) * new_delta)*2
        self.cfg.delta_q_max[4:6] = np.deg2rad(np.array([1,1], dtype=np.float64) * new_delta)*4