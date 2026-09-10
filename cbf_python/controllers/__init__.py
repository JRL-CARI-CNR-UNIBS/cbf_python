"""
Control Barrier Function (CBF) and task controllers package.
"""

from cbf_python.controllers.base_optimal_controller import (
    BCFOptimalController,
    ControllerConfig,
)
from cbf_python.controllers.pid_cbf_controller import UR10CBFController
from cbf_python.controllers.polynomial_controller import (
    PolynomialOptimalController,
    PolynomialControllerConfig,
    compute_generic_lambda,
)
from cbf_python.controllers.gaussian_controller import (
    GaussianController,
    GaussianControllerConfig,
    GaussianSet,
)
from cbf_python.controllers.velocity_scaling import (
    compute_velocity_scaling_for_human_proximity,
)

__all__ = [
    "BCFOptimalController",
    "ControllerConfig",
    "UR10CBFController",
    "PolynomialOptimalController",
    "PolynomialControllerConfig",
    "compute_generic_lambda",
    "GaussianController",
    "GaussianControllerConfig",
    "GaussianSet",
    "compute_velocity_scaling_for_human_proximity",
]
