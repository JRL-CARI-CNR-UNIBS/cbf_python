from scipy.stats import gaussian_kde

from Controller.optimal_cbf_task_controller import BCFOptimalController, ControllerConfig
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Dict

@dataclass
class GaussianSet:
    means: Dict[str, float] = field(
        default_factory=lambda: {"h": 0.0, "d": 0.0, "v_rel": 0.0}
    )
    covariance: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 0.01
    )
    inv_covariance: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 0.01
    )

    # I tipi primitivi immutabili (float, int) non hanno bisogno del default_factory
    norm_const: float = 0.0

    lambda_ref: Dict[str, float] = field(
        default_factory=lambda: {
            "pos": 0.0,
            "vel": 0.0,
            "acc": 0.0,
            "scaling": 0.0,
            "gamma": 0.0
        }
    )

    def __str__(self) -> str:
        """Formats a single GaussianSet, making 2D matrices easy to read."""

        def fmt_arr(arr: np.ndarray) -> str:
            # Formats matrices nicely with consistent spacing and indentation
            formatted = np.array2string(arr, precision=4, suppress_small=True, separator=', ')
            return formatted.replace('\n', '\n      ')

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
    gaussian_sets: list[GaussianSet] = field(default_factory=list)
    n_gaussian_sets: int = 0

    def __str__(self) -> str:
        # 1. Get the base configuration string from the parent class
        base_str = super().__str__()

        # 2. Optionally, update the title to reflect the child class
        base_str = base_str.replace("ControllerConfig:", "GaussianControllerConfig:")

        # 3. Build the Gaussian-specific string
        gauss_str = f"\n\n  -- Gaussian Sets (Total: {self.n_gaussian_sets}) --"

        if not self.gaussian_sets:
            gauss_str += "\n  [No Gaussian Sets Defined]"
        else:
            for i, g_set in enumerate(self.gaussian_sets):
                gauss_str += f"\n  Set {i + 1}:\n"
                gauss_str += str(g_set)

        return base_str + gauss_str

    def precompute_gaussian_parameters(self):
        """
        Compute the inverse of covariance matrix and the normalization constant
        """
        for gaussian_set in self.gaussian_sets:
            cov_matrix = gaussian_set.covariance

            # 1. Compute and store the inverse of the covariance matrix (Σ^-1)
            gaussian_set.inv_covariance = np.linalg.inv(cov_matrix)

            # 2. Compute the determinant of the covariance matrix (|Σ|)
            det_cov = np.linalg.det(cov_matrix)

            # 3. Compute and store the full normalization constant
            # Formula: 1 / sqrt((2*pi)^D * |Σ|)
            const = 1.0 / np.sqrt(((2 * np.pi) ** 3) * det_cov)
            gaussian_set.norm_const = const


    def plot_gaussians(self):
        # Creazione della griglia di subplot
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        for gaussian_set in self.gaussian_sets:
            # 1. Estrazione delle medie (unpacking del vettore)
            mu_h, mu_d, mu_v = list(gaussian_set.means.values())

            # 2. Estrazione delle varianze dalla DIAGONALE della matrice di covarianza
            # cov_matrix[riga, colonna]
            var_h = gaussian_set.covariance[0, 0]  # Posizione 0,0
            var_d = gaussian_set.covariance[1, 1]  # Posizione 1,1
            var_v = gaussian_set.covariance[2, 2]  # Posizione 2,2

            # 3. Calcolo delle deviazioni standard (radice quadrata della varianza)
            std_h = np.sqrt(var_h)
            std_d = np.sqrt(var_d)
            std_v = np.sqrt(var_v)

            # 4. Creazione degli assi X (+/- 4 deviazioni standard per vedere la campana)
            x_h = np.linspace(mu_h - 4 * std_h, mu_h + 4 * std_h, 100)
            x_d = np.linspace(mu_d - 4 * std_d, mu_d + 4 * std_d, 100)
            x_v = np.linspace(mu_v - 4 * std_v, mu_v + 4 * std_v, 100)

            # 5. Calcolo delle PDF per ogni punto x
            pdf_h = norm.pdf(x_h, mu_h, std_h)
            pdf_d = norm.pdf(x_d, mu_d, std_d)
            pdf_v = norm.pdf(x_v, mu_v, std_v)

            # --- Plot per h ---
            axs[0].plot(x_h, pdf_h, 'b-', lw=2, label=f'Normale ($\mu$={mu_h:.2f}, $\sigma$={std_h:.2f})')
            axs[0].fill_between(x_h, pdf_h, alpha=0.2, color='blue')  # Colora l'area sotto la curva
            axs[0].set_title('Distribuzione Marginale di $h$')
            axs[0].set_xlabel('Valore di $h$')
            axs[0].set_ylabel('Densità di Probabilità')
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)

            # --- Plot per d ---
            axs[1].plot(x_d, pdf_d, 'g-', lw=2, label=f'Normale ($\mu$={mu_d:.2f}, $\sigma$={std_d:.2f})')
            axs[1].fill_between(x_d, pdf_d, alpha=0.2, color='green')
            axs[1].set_title('Distribuzione Marginale di $d$')
            axs[1].set_xlabel('Valore di $d$')
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)

            # --- Plot per v ---
            axs[2].plot(x_v, pdf_v, 'r-', lw=2, label=f'Normale ($\mu$={mu_v:.2f}, $\sigma$={std_v:.2f})')
            axs[2].fill_between(x_v, pdf_v, alpha=0.2, color='red')
            axs[2].set_title('Distribuzione Marginale di $v$')
            axs[2].set_xlabel('Valore di $v$')
            axs[2].legend()
            axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        # plt.savefig(filename)
        plt.show()

class GaussianController(BCFOptimalController):

    def __init__(self, model_wrapper, cfg: GaussianControllerConfig, useCbf, keypoint_to_log = 7, n_samples = 50):
        super().__init__(model_wrapper, cfg, useCbf, keypoint_to_log)
        self.cfg = cfg
        self.n_samples = n_samples
        self.cycles = 0


    def update_parameters(self, h, d, v_rel):
        """
        Calculate the blended lambda parameters for the current system state in real-time.

        Args:
            h (float): Current h state.
            d (float): Current d state.
            v_rel (float): Current relative velocity state.
        """
        current_state = np.array([h, d, v_rel])
        raw_weights = []

        # Step 1: Calculate the probability density (weight) for each condition
        for gaussian_set in self.cfg.gaussian_sets:
            # Extract the mean vector ensuring the exact same order as current_state: [h, d, v_rel]
            mean_vector = np.array([
                gaussian_set.means["h"],
                gaussian_set.means["d"],
                gaussian_set.means["v_rel"]
            ])

            # Vector difference from current state to the Gaussian mean (x - μ)
            diff = current_state - mean_vector

            # Calculate the exponent using the squared Mahalanobis distance
            # Using the '@' operator for fast matrix multiplication
            exponent = -0.5 * (diff.T @ gaussian_set.inv_covariance @ diff)

            # Final Probability Density Function (PDF) evaluation using precomputed constant
            weight = gaussian_set.norm_const * np.exp(exponent)
            raw_weights.append(weight)

        # Step 2: Normalize the weights so they sum up to 1.0 (100%)
        total_weight = sum(raw_weights)

        # Safety fallback: Prevent division by zero if the state is extremely
        # far from all known conditions (all weights round down to 0.0)
        if total_weight == 0:
            return 0.0, [0.0] * self.cfg.n_gaussian_sets

        normalized_weights = [w / total_weight for w in raw_weights]
        final_lambdas =  {
            "pos": 0.0,
            "vel": 0.0,
            "acc": 0.0,
            "scaling": 0.0,
            "gamma": 0.0
        }

        for val in final_lambdas:
            for i in range(self.cfg.n_gaussian_sets):
                final_lambdas[val] += normalized_weights[i]*self.cfg.gaussian_sets[i].lambda_ref[val]
        self.cfg.lambda_pos = final_lambdas["pos"]
        self.cfg.lambda_vel = final_lambdas["vel"]
        self.cfg.lambda_acc = final_lambdas["acc"]
        self.cfg.lambda_scaling = final_lambdas["scaling"]
        self.cfg.gamma = final_lambdas["gamma"]