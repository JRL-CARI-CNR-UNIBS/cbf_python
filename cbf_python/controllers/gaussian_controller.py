"""
Gaussian mixture parameter blending CBF controller and configuration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from cbf_python.controllers.base_optimal_controller import (
    BCFOptimalController,
    ControllerConfig,
)


@dataclass
class GaussianSet:
    """Represents a Gaussian parameter cluster in state space (h, d, v_rel)."""

    means: Dict[str, float] = field(
        default_factory=lambda: {"h": 0.0, "d": 0.0, "v_rel": 0.0}
    )
    covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.01)
    inv_covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.01)
    norm_const: float = 0.0
    lambda_ref: Dict[str, float] = field(
        default_factory=lambda: {
            "pos": 0.0,
            "vel": 0.0,
            "acc": 0.0,
            "scaling": 0.0,
            "gamma": 0.0,
        }
    )

    def __str__(self) -> str:
        """Format GaussianSet properties into a readable string."""
        def fmt_arr(arr: np.ndarray) -> str:
            formatted = np.array2string(arr, precision=4, suppress_small=True, separator=", ")
            return formatted.replace("\n", "\n      ")

        means_str = ", ".join([f"'{k}': {v:.4g}" for k, v in self.means.items()])
        lambdas_str = ", ".join([f"'{k}': {v:.4g}" for k, v in self.lambda_ref.items()])

        return (
            f"    means         : {{ {means_str} }}\n"
            f"    norm_const    : {self.norm_const:.4g}\n"
            f"    lambda_ref    : {{ {lambdas_str} }}\n"
            f"    covariance    :\n      {fmt_arr(self.covariance)}\n"
            f"    inv_covariance:\n      {fmt_arr(self.inv_covariance)}"
        )


@dataclass
class GaussianControllerConfig(ControllerConfig):
    """Configuration for Gaussian mixture parameter blending."""

    gaussian_sets: List[GaussianSet] = field(default_factory=list)
    n_gaussian_sets: int = 0

    def __str__(self) -> str:
        base_str = super().__str__().replace("ControllerConfig:", "GaussianControllerConfig:")
        gauss_str = f"\n\n  -- Gaussian Sets (Total: {self.n_gaussian_sets}) --"
        if not self.gaussian_sets:
            gauss_str += "\n  [No Gaussian Sets Defined]"
        else:
            for i, g_set in enumerate(self.gaussian_sets):
                gauss_str += f"\n  Set {i + 1}:\n{g_set}"
        return base_str + gauss_str

    def precompute_gaussian_parameters(self) -> None:
        """Compute inverse covariance matrices and multivariate normal normalization constants."""
        for g_set in self.gaussian_sets:
            cov = g_set.covariance
            g_set.inv_covariance = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            # 1 / sqrt((2*pi)^3 * |Sigma|)
            g_set.norm_const = 1.0 / np.sqrt(((2.0 * np.pi) ** 3) * det_cov)

    def plot_gaussians(self) -> None:
        """Plot 1D marginal distributions for h, d, and v_rel for each set."""
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        for g_set in self.gaussian_sets:
            mu_h, mu_d, mu_v = list(g_set.means.values())
            var_h = g_set.covariance[0, 0]
            var_d = g_set.covariance[1, 1]
            var_v = g_set.covariance[2, 2]

            std_h = np.sqrt(var_h)
            std_d = np.sqrt(var_d)
            std_v = np.sqrt(var_v)

            x_h = np.linspace(mu_h - 4 * std_h, mu_h + 4 * std_h, 100)
            x_d = np.linspace(mu_d - 4 * std_d, mu_d + 4 * std_d, 100)
            x_v = np.linspace(mu_v - 4 * std_v, mu_v + 4 * std_v, 100)

            pdf_h = norm.pdf(x_h, mu_h, std_h)
            pdf_d = norm.pdf(x_d, mu_d, std_d)
            pdf_v = norm.pdf(x_v, mu_v, std_v)

            axs[0].plot(x_h, pdf_h, "b-", lw=2, label=f"N(mu={mu_h:.2f}, sigma={std_h:.2f})")
            axs[0].fill_between(x_h, pdf_h, alpha=0.2, color="blue")
            axs[0].set_title("Marginal Distribution: h")
            axs[0].set_xlabel("h [m]")
            axs[0].set_ylabel("PDF")
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)

            axs[1].plot(x_d, pdf_d, "g-", lw=2, label=f"N(mu={mu_d:.2f}, sigma={std_d:.2f})")
            axs[1].fill_between(x_d, pdf_d, alpha=0.2, color="green")
            axs[1].set_title("Marginal Distribution: d")
            axs[1].set_xlabel("d [m]")
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)

            axs[2].plot(x_v, pdf_v, "r-", lw=2, label=f"N(mu={mu_v:.2f}, sigma={std_v:.2f})")
            axs[2].fill_between(x_v, pdf_v, alpha=0.2, color="red")
            axs[2].set_title("Marginal Distribution: v")
            axs[2].set_xlabel("v [m/s]")
            axs[2].legend()
            axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class GaussianController(BCFOptimalController):
    """Optimal task controller with Gaussian mixture-weighted parameter interpolation."""

    def __init__(
        self,
        model_wrapper: Any,
        cfg: GaussianControllerConfig,
        useCbf: bool = True,
        keypoint_to_log: int = 7,
        n_samples: int = 50,
    ) -> None:
        super().__init__(model_wrapper, cfg, useCbf, keypoint_to_log)
        self.cfg: GaussianControllerConfig = cfg
        self.n_samples = n_samples
        self.cycles = 0

    def update_parameters(self, h: float, d: float, v_rel: float) -> None:
        """Evaluate Mahalanobis distance weights and blend controller lambdas dynamically."""
        current_state = np.array([h, d, v_rel])
        raw_weights = []

        for g_set in self.cfg.gaussian_sets:
            mean_vector = np.array([
                g_set.means["h"],
                g_set.means["d"],
                g_set.means["v_rel"],
            ])
            diff = current_state - mean_vector
            exponent = -0.5 * float(diff.T @ g_set.inv_covariance @ diff)
            weight = g_set.norm_const * np.exp(exponent)
            raw_weights.append(weight)

        total_weight = sum(raw_weights)
        if total_weight == 0.0 or not np.isfinite(total_weight):
            return

        normalized_weights = [w / total_weight for w in raw_weights]
        final_lambdas = {
            "pos": 0.0,
            "vel": 0.0,
            "acc": 0.0,
            "scaling": 0.0,
            "gamma": 0.0,
        }

        for val in final_lambdas:
            for i in range(self.cfg.n_gaussian_sets):
                final_lambdas[val] += normalized_weights[i] * self.cfg.gaussian_sets[i].lambda_ref[val]

        self.cfg.lambda_pos = final_lambdas["pos"]
        self.cfg.lambda_vel = final_lambdas["vel"]
        self.cfg.lambda_acc = final_lambdas["acc"]
        self.cfg.lambda_scaling = final_lambdas["scaling"]
        self.cfg.gamma = final_lambdas["gamma"]
