from Controller.optimal_cbf_task_controller import BCFOptimalController, ControllerConfig
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
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

    lambda_f_pos : float = 0.0
    lambda_f_vel : float = 0.0
    lambda_f_acc : float= 0.0
    lambda_f_scaling : float = 0.0
    gamma_f : float = 0.0

    n_pos : float = 0.0
    n_vel : float = 0.0
    n_acc : float= 0.0
    n_scaling : float = 0.0
    n_gamma : float = 0.0

    m_pos : float = 0.0
    m_vel : float = 0.0
    m_acc : float= 0.0
    m_scaling : float = 0.0
    m_gamma : float = 0.0

    w_pos : float = 0.0
    w_vel : float = 0.0
    w_acc : float= 0.0
    w_scaling : float = 0.0
    w_gamma : float = 0.0

    h_t : float = 0.0
    polynomial_dict = {"pos": [], "vel": [], "acc": [], "scaling": [], "gamma": []}

    def generate_poly_dict(self):
        self.polynomial_dict["pos"] = [self.lambda_0_pos, self.lambda_f_pos, self.n_pos, self.m_pos, self.w_pos]
        self.polynomial_dict["vel"] = [self.lambda_0_vel, self.lambda_f_vel, self.n_vel, self.m_vel, self.w_vel]
        self.polynomial_dict["acc"] = [self.lambda_0_acc, self.lambda_f_acc, self.n_acc, self.m_acc, self.w_acc]
        self.polynomial_dict["scaling"] = [self.lambda_0_scaling, self.lambda_f_scaling, self.n_scaling, self.m_scaling, self.w_scaling]
        self.polynomial_dict["gamma"] = [self.gamma_0, self.gamma_f, self.n_gamma, self.m_gamma, self.w_gamma]

    def __str__(self):
        # 1. Get the base configuration string from the parent class
        base_str = super().__str__()

        # 2. Optionally, update the title to reflect the child class
        base_str = base_str.replace("ControllerConfig:", "PolynomialControllerConfig:")

        # 3. Format the specific attributes of this child class cleanly
        poly_str = (
            f"\nPolynomial Parameters (h_t = {self.h_t}):\n"
            f"  Position: [lambda_0: {self.lambda_0_pos}, lambda_f: {self.lambda_f_pos}, n: {self.n_pos}, m: {self.m_pos}, w: {self.w_pos}]\n"
            f"  Velocity: [lambda_0: {self.lambda_0_vel}, lambda_f: {self.lambda_f_vel}, n: {self.n_vel}, m: {self.m_vel}, w: {self.w_vel}]\n"
            f"  Accel:    [lambda_0: {self.lambda_0_acc}, lambda_f: {self.lambda_f_acc}, n: {self.n_acc}, m: {self.m_acc}, w: {self.w_acc}]\n"
            f"  Scaling:  [lambda_0: {self.lambda_0_scaling}, lambda_f: {self.lambda_f_scaling}, n: {self.n_scaling}, m: {self.m_scaling}, w: {self.w_scaling}]\n"
            f"  Gamma:    [gamma_0: {self.gamma_0}, gamma_f: {self.gamma_f}, n: {self.n_gamma}, m: {self.m_gamma}, w: {self.w_gamma}]\n"
        )

        return base_str + poly_str

    def plot_lambdas(self):
        """Plots the piecewise polynomial evolution of all parameters as a function of h."""
        self.generate_poly_dict()
        ht = self.h_t if self.h_t != 0.0 else 1.0

        # Range extended slightly to visualize the flat regions clearly
        h_vals = np.linspace(-0.2, ht + 0.2, 50000)

        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        axes = axes.flatten()
        categories = ["pos", "vel", "acc", "scaling", "gamma"]

        for i, cat in enumerate(categories):
            l0, lf, n, m, w = self.polynomial_dict[cat]

            # 1. Define the piecewise conditions
            conditions = [
                h_vals < 0.0,
                h_vals >= ht,
                (h_vals >= 0.0) & (h_vals < ht)
            ]

            # 2. Calculate the polynomial values (using np.clip to prevent any edge-case warnings)
            safe_h = np.clip(h_vals, 0.0, ht)
            base = safe_h / ht
            poly_vals = l0 + (lf - l0) * (w * (base ** n) + (1 - w) * (base ** m))

            # 3. Define the outputs corresponding to each condition
            choices = [
                l0,  # Output if h < 0.0
                lf,  # Output if h >= ht
                poly_vals  # Output if 0 <= h < ht
            ]

            # Apply the piecewise law
            y_vals = np.select(conditions, choices)

            # Plotting
            axes[i].plot(h_vals, y_vals, color='#1f77b4', linewidth=2.5)
            axes[i].axvline(0, color='red', linestyle='--', alpha=0.6, label='h=0 (Boundary)')
            axes[i].axvline(ht, color='green', linestyle='--', alpha=0.6, label='h=h_t (Target)')

            axes[i].set_title(f"Parameter: {cat.upper()}", fontweight='bold')
            axes[i].set_xlabel("h (CBF Value)")
            axes[i].set_ylabel("Weight Value")
            axes[i].grid(True, linestyle=':', alpha=0.7)
            axes[i].legend(loc="best", fontsize="small")

        axes[5].set_visible(False)
        plt.tight_layout()

    def check_config_integrity(self):
        """ Function to check that all lambdas are positive and there is no jump in the interval [0, h_t]"""

        self.generate_poly_dict()
        
        for index in self.polynomial_dict.keys():
            l_0 = self.polynomial_dict[index][0]
            l_f = self.polynomial_dict[index][1]
            n = self.polynomial_dict[index][2]
            m = self.polynomial_dict[index][3]
            w = self.polynomial_dict[index][4]

            MAX_JUMP_TOLERANCE = 1e-3*abs(l_0)

            # Relaxed slope constraint (Prevent LARGE vertical walls at h=0)
            if n < 1.0:
                # Calculate the approximate magnitude of the instant step
                jump_magnitude = abs((l_f - l_0) * w)
                if jump_magnitude > MAX_JUMP_TOLERANCE:
                    return {"res" : False,
                            "cause": "jump",
                            "magnitude": jump_magnitude,
                            "tolerance": MAX_JUMP_TOLERANCE
                            }# The jump is too large, reject it.

            # Prevent ZeroDivisionError (constant weight)
            if l_f == l_0:
                if l_0 < 0:
                    return {"res": False,
                            "cause": "negative weight",
                            "magnitude": l_0
                            } # The constant weight is already negative!
                continue # Everything is fine, proceed to the next parameter

            # Safety check on exponents
            if m == n:
                continue # Avoid zero division if m and n are equal
                
            # Upper limit for w beyond which an overshoot occurs
            w_upper_limit = m / (m - n)
            
            # Calculate the extremum ONLY if w creates an undershoot or an overshoot
            if w < 0 or w > w_upper_limit:
                base = (w * n) / ((w - 1) * m)
                
                # Prevent Math Domain Error (extra safety)
                if base > 0:
                    y_ext = w * ((m - n) / m) * (base ** (n / (m - n)))
                    
                    # The comparison term is IDENTICAL in both cases
                    comparison_term = -l_0 / (l_f - l_0)
                    
                    if l_f > l_0:
                        # Increasing transition: risk of undershoot
                        if y_ext < comparison_term:
                            return {"res": False,
                                    "cause": "Increasing transition: risk of undershoot",
                                    }
                    else:
                        # Decreasing transition: risk of overshoot
                        if y_ext > comparison_term:
                            return {"res": False,
                                    "cause": "Decreasing transition: risk of overshoot",
                                    }
                            
        return {"res":True}
    
    def normalize_parameters(self):
        """
        Normalizes the polynomial parameters to ensure n <= m for all tasks.
        If n > m, it swaps the exponents and inverts the weight (w = 1 - w).
        The resulting mathematical curve remains strictly identical.
        """
        categories = ["pos", "vel", "acc", "scaling", "gamma"]
        
        for cat in categories:
            # Retrieve current values
            n = getattr(self, f"n_{cat}")
            m = getattr(self, f"m_{cat}")
            w = getattr(self, f"w_{cat}")
            
            # Check if normalization is needed
            if n > m:
                # Apply normalization rules
                setattr(self, f"n_{cat}", m)
                setattr(self, f"m_{cat}", n)
                setattr(self, f"w_{cat}", 1.0 - w)
                
        # Update the internal dictionary with the newly normalized values
        self.generate_poly_dict()



class PolynomialOptimalController(BCFOptimalController):

    def __init__(self, model_wrapper, cfg: PolynomialControllerConfig, useCbf, keypoint_to_log = 7):
        super().__init__(model_wrapper, cfg, useCbf, keypoint_to_log)
        self.cfg = cfg
        self.cfg.generate_poly_dict()
        

    def update_parameters(self, h, d, v_rel):

        self.cfg.lambda_pos = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["pos"])
        self.cfg.lambda_vel = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["vel"])
        self.cfg.lambda_acc = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["acc"])
        self.cfg.lambda_scaling = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["scaling"])
        self.cfg.gamma = compute_generic_lambda(h, self.cfg.h_t, self.cfg.polynomial_dict["gamma"])
