"""
Polynomial dynamic parameter CBF controller and configuration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

from cbf_python.controllers.base_optimal_controller import (
    BCFOptimalController,
    ControllerConfig,
)


def compute_generic_lambda(h: float, ht: float, params: Sequence[float]) -> float:
    """Compute generic piece-wise polynomial lambda parameter as a function of barrier margin h.

    Parameters
    ----------
    h : float
        Current barrier certificate value.
    ht : float
        Transition threshold margin.
    params : sequence of float
        [lambda_0, lambda_f, n, m, w]
    """
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
        norm_h = h / ht if ht > 0.0 else 0.0
        return lambda_0 + (lambda_f - lambda_0) * (w * (norm_h ** n) + (1.0 - w) * (norm_h ** m))


@dataclass
class PolynomialControllerConfig(ControllerConfig):
    """Configuration for polynomial parameter modulation."""

    lambda_0_pos: float = 0.0
    lambda_0_vel: float = 0.0
    lambda_0_acc: float = 0.0
    lambda_0_scaling: float = 0.0
    gamma_0: float = 0.0

    lambda_f_pos: float = 0.0
    lambda_f_vel: float = 0.0
    lambda_f_acc: float = 0.0
    lambda_f_scaling: float = 0.0
    gamma_f: float = 0.0

    n_pos: float = 0.0
    n_vel: float = 0.0
    n_acc: float = 0.0
    n_scaling: float = 0.0
    n_gamma: float = 0.0

    m_pos: float = 0.0
    m_vel: float = 0.0
    m_acc: float = 0.0
    m_scaling: float = 0.0
    m_gamma: float = 0.0

    w_pos: float = 0.0
    w_vel: float = 0.0
    w_acc: float = 0.0
    w_scaling: float = 0.0
    w_gamma: float = 0.0

    h_t: float = 0.0
    polynomial_dict: Dict[str, List[float]] = field(
        default_factory=lambda: {"pos": [], "vel": [], "acc": [], "scaling": [], "gamma": []}
    )

    def generate_poly_dict(self) -> None:
        """Assemble dictionary of polynomial parameter lists for each weight category."""
        self.polynomial_dict["pos"] = [self.lambda_0_pos, self.lambda_f_pos, self.n_pos, self.m_pos, self.w_pos]
        self.polynomial_dict["vel"] = [self.lambda_0_vel, self.lambda_f_vel, self.n_vel, self.m_vel, self.w_vel]
        self.polynomial_dict["acc"] = [self.lambda_0_acc, self.lambda_f_acc, self.n_acc, self.m_acc, self.w_acc]
        self.polynomial_dict["scaling"] = [self.lambda_0_scaling, self.lambda_f_scaling, self.n_scaling, self.m_scaling, self.w_scaling]
        self.polynomial_dict["gamma"] = [self.gamma_0, self.gamma_f, self.n_gamma, self.n_gamma, self.w_gamma]

    def __str__(self) -> str:
        base_str = super().__str__().replace("ControllerConfig:", "PolynomialControllerConfig:")
        poly_str = (
            f"\nPolynomial Parameters (h_t = {self.h_t}):\n"
            f"  Position: [lambda_0: {self.lambda_0_pos}, lambda_f: {self.lambda_f_pos}, n: {self.n_pos}, m: {self.m_pos}, w: {self.w_pos}]\n"
            f"  Velocity: [lambda_0: {self.lambda_0_vel}, lambda_f: {self.lambda_f_vel}, n: {self.n_vel}, m: {self.m_vel}, w: {self.w_vel}]\n"
            f"  Accel:    [lambda_0: {self.lambda_0_acc}, lambda_f: {self.lambda_f_acc}, n: {self.n_acc}, m: {self.m_acc}, w: {self.w_acc}]\n"
            f"  Scaling:  [lambda_0: {self.lambda_0_scaling}, lambda_f: {self.lambda_f_scaling}, n: {self.n_scaling}, m: {self.m_scaling}, w: {self.w_scaling}]\n"
            f"  Gamma:    [gamma_0: {self.gamma_0}, gamma_f: {self.gamma_f}, n: {self.n_gamma}, m: {self.m_gamma}, w: {self.w_gamma}]\n"
        )
        return base_str + poly_str

    def plot_lambdas(self, save_fig: bool = False, filename: str = "Controller_weights_plot.pdf") -> None:
        """Plot the piecewise polynomial evolution of all parameters as a function of h."""
        self.generate_poly_dict()
        ht = self.h_t if self.h_t != 0.0 else 1.0
        h_vals = np.linspace(-0.2, ht + 0.2, 5000)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        categories = ["pos", "vel", "acc", "scaling"]

        for i, cat in enumerate(categories):
            l0, lf, n, m, w = self.polynomial_dict[cat]
            conditions = [h_vals < 0.0, h_vals >= ht, (h_vals >= 0.0) & (h_vals < ht)]
            safe_h = np.clip(h_vals, 0.0, ht)
            base = safe_h / ht
            poly_vals = l0 + (lf - l0) * (w * (base ** n) + (1.0 - w) * (base ** m))
            choices = [l0, lf, poly_vals]
            y_vals = np.select(conditions, choices)

            axes[i].plot(h_vals, y_vals, color="#1f77b4", linewidth=2.5)
            axes[i].axvline(0, color="red", linestyle="--", alpha=0.6, label="$h_{inf} = 0$")
            axes[i].axvline(ht, color="green", linestyle="--", alpha=0.6, label=f"$h_{{sup}} = {ht}$")
            axes[i].set_title(f"Parameter: $\\lambda_{{{cat}}}$", fontweight="bold", fontsize=16)
            axes[i].set_xlabel("h [m]", fontsize=12)
            axes[i].set_ylabel("Weight Value", fontsize=12)
            axes[i].grid(True, linestyle=":", alpha=0.7)
            axes[i].legend(loc="best", fontsize=10)

        plt.tight_layout()
        if save_fig:
            plt.savefig(filename)
        plt.show()

    def check_config_integrity(self) -> Dict[str, Any]:
        """Verify that all lambdas remain positive and devoid of catastrophic jumps."""
        self.generate_poly_dict()

        for index in self.polynomial_dict.keys():
            l_0 = self.polynomial_dict[index][0]
            l_f = self.polynomial_dict[index][1]
            n = self.polynomial_dict[index][2]
            m = self.polynomial_dict[index][3]
            w = self.polynomial_dict[index][4]

            max_jump_tolerance = max(1e-3 * abs(l_0), 1e-10)

            if n < 1.0:
                jump_magnitude = abs((l_f - l_0) * w)
                if jump_magnitude > max_jump_tolerance:
                    return {
                        "res": False,
                        "cause": "jump",
                        "magnitude": jump_magnitude,
                        "tolerance": max_jump_tolerance,
                    }

            if l_f == l_0:
                if l_0 < 0:
                    return {"res": False, "cause": "negative weight", "magnitude": l_0}
                continue

            if m == n:
                continue

            w_upper_limit = m / (m - n) if (m - n) != 0 else 0.0
            if w < 0 or w > w_upper_limit:
                denom = (w - 1.0) * m
                if denom != 0:
                    base = (w * n) / denom
                    if base > 0:
                        y_ext = w * ((m - n) / m) * (base ** (n / (m - n)))
                        comparison_term = -l_0 / (l_f - l_0)
                        if l_f > l_0 and y_ext < comparison_term:
                            return {"res": False, "cause": "Increasing transition: risk of undershoot"}
                        elif l_f <= l_0 and y_ext > comparison_term:
                            return {"res": False, "cause": "Decreasing transition: risk of overshoot"}

        return {"res": True}

    def normalize_parameters(self) -> None:
        """Normalize exponents to ensure n <= m, preserving identical mathematical curves."""
        categories = ["pos", "vel", "acc", "scaling", "gamma"]
        for cat in categories:
            n = getattr(self, f"n_{cat}")
            m = getattr(self, f"m_{cat}")
            w = getattr(self, f"w_{cat}")
            if n > m:
                setattr(self, f"n_{cat}", m)
                setattr(self, f"m_{cat}", n)
                setattr(self, f"w_{cat}", 1.0 - w)
        self.generate_poly_dict()


class PolynomialOptimalController(BCFOptimalController):
    """Optimal task CBF controller with piecewise polynomial weights modulated by margin h."""

    def __init__(
        self,
        model_wrapper: Any,
        cfg: PolynomialControllerConfig,
        useCbf: bool = True,
        keypoint_to_log: int = 7,
    ) -> None:
        super().__init__(model_wrapper, cfg, useCbf, keypoint_to_log)
        self.cfg: PolynomialControllerConfig = cfg
        self.cfg.generate_poly_dict()

    def update_parameters(self, h: float, d: float, v_rel: float) -> None:
        """Modulate controller weights in real time according to current barrier margin h."""
        self.cfg.lambda_pos = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["pos"])
        self.cfg.lambda_vel = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["vel"])
        self.cfg.lambda_acc = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["acc"])
        self.cfg.lambda_scaling = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["scaling"])
        self.cfg.gamma = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["gamma"])
