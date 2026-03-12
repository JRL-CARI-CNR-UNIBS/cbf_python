from Controller.optimal_cbf_task_controller import BCFOptimalController, ControllerConfig
from dataclasses import dataclass, field
import numpy as np
def compute_generic_lambda(h, ht, params):
    lambda_0 = params[0]
    lambda_f = params[1]
    n = params[2]
    m = params[3]
    w = params[4]
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
    polynomial_dict = {"pos": [], "vel": [], "acc": [], "scaling": [], "gamma": []}

    def generate_poly_dict(self):
        self.polynomial_dict["pos"] = [self.lambda_0_pos, self.lambda_f_pos, self.n_pos, self.m_pos, self.w_pos]
        self.polynomial_dict["vel"] = [self.lambda_0_vel, self.lambda_f_vel, self.n_vel, self.m_vel, self.w_vel]
        self.polynomial_dict["acc"] = [self.lambda_0_acc, self.lambda_f_acc, self.n_acc, self.m_acc, self.w_acc]
        self.polynomial_dict["scaling"] = [self.lambda_0_scaling, self.lambda_f_scaling, self.n_scaling, self.m_scaling, self.w_scaling]
        self.polynomial_dict["gamma"] = [self.gamma_0, self.gamma_f, self.n_gamma, self.m_gamma, self.w_gamma]

class PolynomialOptimalController(BCFOptimalController):

    def __init__(self, model_wrapper, cfg: PolynomialControllerConfig, useCbf, keypoint_to_log = 7):
        super().__init__(model_wrapper, cfg, useCbf, keypoint_to_log)
        self.cfg = cfg
        self.cfg.generate_poly_dict()
        

    def update_parameters(self, h):

        self.cfg.lambda_pos = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["pos"])
        self.cfg.lambda_vel = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["vel"])
        self.cfg.lambda_acc = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["acc"])
        self.cfg.lambda_scaling = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["scaling"])
        self.cfg.gamma = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["gamma"])
