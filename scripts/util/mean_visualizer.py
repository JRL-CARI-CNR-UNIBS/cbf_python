from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


class StochasticCBFVisualizer:
    """
    Online accumulator and statistical analyzer for barrier functions and distance metrics.
    """

    def __init__(self, n: int = 50):
        self.v_mean: Optional[float] = None
        self.d_mean: Optional[float] = None
        self.h_mean: Optional[float] = None
        self.cov_matrix: Optional[np.ndarray] = None
        self.n = n
        self.cycles = 0
        self.h_window = np.zeros(n)
        self.h_vec: List[float] = []
        self.d_vec: List[float] = []
        self.v_vec: List[float] = []
        self.time_vec: List[float] = []
        self.lambda_vec: List[float] = []

    def update_vectors(self, h: float, d: float, v_rel: float, t: float) -> None:
        """Appends current iteration measurements."""
        self.h_vec.append(float(h))
        self.d_vec.append(float(d))
        self.v_vec.append(float(v_rel))
        self.time_vec.append(float(t))

    def compute_mean_cov(self, print_val: bool = False) -> Tuple[np.ndarray, float, float, float]:
        """
        Computes sample mean and 3x3 covariance matrix across [h, d, v_rel].

        Returns:
            cov_matrix : np.ndarray (3, 3)
            h_mean     : float
            d_mean     : float
            v_mean     : float
        """
        data_matrix = np.vstack((self.h_vec, self.d_vec, self.v_vec))
        self.cov_matrix = np.cov(data_matrix)
        self.h_mean = float(np.mean(self.h_vec))
        self.d_mean = float(np.mean(self.d_vec))
        self.v_mean = float(np.mean(self.v_vec))

        if print_val:
            var_h = float(np.std(self.h_vec, ddof=1))
            var_d = float(np.std(self.d_vec, ddof=1))
            var_v = float(np.std(self.v_vec, ddof=1))
            print("--- Data Means ---")
            print(f"h: {self.h_mean:.4f}, d: {self.d_mean:.4f}, v: {self.v_mean:.4f}")
            print("\n--- Covariance Matrix (Sigma) ---")
            print(np.round(self.cov_matrix, 4))
            print("\n--- Standard Deviations ---")
            print(f"std(h): {var_h:.4f}, std(d): {var_d:.4f}, std(v): {var_v:.4f}")

        return self.cov_matrix, self.h_mean, self.d_mean, self.v_mean