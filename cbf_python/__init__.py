"""
CBF Python Package: Modular, high-performance Control Barrier Functions for robot manipulators.

This package provides:
    - Optimization-based motion controllers enforcing Speed and Separation Monitoring (SSM)
    - High-performance Numba Lie-derivative and QP assembly kernels
    - Real-time parameter adaptation via piecewise polynomial and Gaussian mixture blending
    - Robot command and sensor bridges for both hardware execution and simulated human tracking
"""

from __future__ import annotations
import os
import sys
from pathlib import Path


def _configure_cmeel_environment() -> None:
    """Dynamically configure sys.path and LD_LIBRARY_PATH for CMEEL-packaged Pinocchio if needed.

    Checks first if Pinocchio can already be loaded from the current environment (e.g. system
    ROS 2 packages). Only activates and configures the CMEEL bundle when Pinocchio cannot
    be imported otherwise.
    """
    # 1. Fast check: if Pinocchio is already functional in the active environment, do nothing.
    try:
        import pinocchio  # noqa: F401
        return
    except Exception:
        pass

    # 2. Discover potential virtualenv prefixes dynamically (no hardcoded user paths)
    candidate_prefixes: list[Path] = [Path(sys.prefix)]

    if "VIRTUAL_ENV" in os.environ:
        candidate_prefixes.append(Path(os.environ["VIRTUAL_ENV"]))

    # Look for virtualenvs in parent directories if present
    curr = Path(__file__).resolve().parent
    for _ in range(4):
        curr = curr.parent
        for venv_name in ["cbf_venv", "venv", ".venv"]:
            v_path = curr / venv_name
            if v_path.exists() and v_path not in candidate_prefixes:
                candidate_prefixes.append(v_path)

    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"

    for prefix in candidate_prefixes:
        cmeel_root = prefix / "lib" / py_ver / "site-packages" / "cmeel.prefix"
        if not cmeel_root.exists():
            cmeel_root = prefix / "lib" / "site-packages" / "cmeel.prefix"

        if cmeel_root.exists():
            cmeel_py = cmeel_root / "lib" / py_ver / "site-packages"
            if cmeel_py.exists():
                cmeel_py_str = str(cmeel_py)
                if cmeel_py_str in sys.path:
                    sys.path.remove(cmeel_py_str)
                sys.path.insert(0, cmeel_py_str)

            cmeel_lib = cmeel_root / "lib"
            if cmeel_lib.exists():
                cmeel_lib_str = str(cmeel_lib)
                ld_parts = os.environ.get("LD_LIBRARY_PATH", "").split(":")
                if ld_parts[0] != cmeel_lib_str and not os.environ.get("_CMEEL_REEXEC"):
                    os.environ["LD_LIBRARY_PATH"] = f"{cmeel_lib_str}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
                    os.environ["_CMEEL_REEXEC"] = "1"
                    try:
                        if hasattr(sys, "orig_argv") and sys.orig_argv:
                            os.execv(sys.orig_argv[0], sys.orig_argv)
                        else:
                            os.execv(sys.executable, [sys.executable] + sys.argv)
                    except Exception:
                        pass
            break


_configure_cmeel_environment()

__version__ = "0.1.0"

# Public API convenience imports
from cbf_python.controllers.base_optimal_controller import (
    BCFOptimalController,
    ControllerConfig,
)
from cbf_python.controllers.polynomial_controller import (
    PolynomialOptimalController,
    PolynomialControllerConfig,
)
from cbf_python.controllers.gaussian_controller import (
    GaussianController,
    GaussianControllerConfig,
    GaussianSet,
)
from cbf_python.controllers.pid_cbf_controller import UR10CBFController
from cbf_python.bridges.fake_bridge import FakeCommandBridge
from cbf_python.bridges.joint_bridge import JointStateCommandBridge
from cbf_python.utils.config_loader import resolve_path, load_yaml

__all__ = [
    "BCFOptimalController",
    "ControllerConfig",
    "PolynomialOptimalController",
    "PolynomialControllerConfig",
    "GaussianController",
    "GaussianControllerConfig",
    "GaussianSet",
    "UR10CBFController",
    "FakeCommandBridge",
    "JointStateCommandBridge",
    "resolve_path",
    "load_yaml",
    "__version__",
]
