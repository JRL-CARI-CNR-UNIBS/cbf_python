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

@dataclass
class StocasticalControllerConfig(PolynomialControllerConfig):

    n: int = 50
    cv_tol: float = 0.25
    p : float = 2
    k_min: float = 1e-04
    k_max: float = 1.0
    sigma_tol = 0.0

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

class StocasticalOptimalController(BCFOptimalController):
    def __init__(self, model_wrapper, cfg: StocasticalControllerConfig, useCbf, keypoint_to_log = 7):
        super().__init__(model_wrapper, cfg, useCbf, keypoint_to_log)
        self.cfg = cfg
        self.cycles = 0
        self.h_vec = np.zeros(self.cfg.n)
        self.h_mean = 0.0
        self.h_std = 0.0
        self.cfg.generate_poly_dict()

    def update_mean_and_std(self, h):
        self.h_vec = np.roll(self.h_vec, -1)
        self.h_vec[-1] = h
        if self.cycles < self.cfg.n:
            self.h_mean = np.mean(self.h_vec[-self.cycles:])
            self.h_std = np.std(self.h_vec[-self.cycles:])
            self.cycles += 1
        else:
            self.h_mean = np.mean(self.h_vec)
            self.h_std = np.std(self.h_vec)

    def update_parameters(self, h):
        self.update_mean_and_std(h)
        epsilon = 1e-06
        cv_squared =  self.h_std / (self.h_mean ** 2 + epsilon)
        cv_tol_squared = self.cfg.cv_tol ** 2
        argument = 0.5 * (cv_squared / cv_tol_squared) ** self.cfg.p
        # sigma_squared =  self.h_std ** 2
        # sigma_tol_squared = self.cfg.sigma_tol ** 2
        # argument = 0.5 * (sigma_squared / sigma_tol_squared) ** self.cfg.p
        k_gain = self.cfg.k_min + (self.cfg.k_max - self.cfg.k_min) * np.exp(-argument)

        lambda_pos_ref = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["pos"])
        self.cfg.lambda_pos = self.cfg.lambda_pos + k_gain * (lambda_pos_ref - self.cfg.lambda_pos)

        lambda_vel_ref = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["vel"])
        self.cfg.lambda_vel = self.cfg.lambda_vel + k_gain * (lambda_vel_ref - self.cfg.lambda_vel)

        lambda_acc_ref = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["acc"])
        self.cfg.lambda_acc = self.cfg.lambda_acc + k_gain * (lambda_acc_ref - self.cfg.lambda_acc)

        lambda_scaling_ref = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["scaling"])
        self.cfg.lambda_scaling = self.cfg.lambda_scaling + k_gain * (lambda_scaling_ref - self.cfg.lambda_scaling)

        lambda_gamma_ref = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["gamma"])
        self.cfg.gamma = self.cfg.gamma + k_gain * (lambda_gamma_ref - self.cfg.gamma)
