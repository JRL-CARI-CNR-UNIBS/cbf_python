"""
Configuration loader for YAML parameters and structured dataclasses.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
import numpy as np

from cbf_python.controllers.base_optimal_controller import ControllerConfig
from cbf_python.controllers.polynomial_controller import PolynomialControllerConfig
from cbf_python.controllers.gaussian_controller import GaussianControllerConfig, GaussianSet


def get_package_root() -> Path:
    """Return the absolute path to the cbf_python package root directory."""
    # Current file is in cbf_python/cbf_python/utils/config_loader.py
    # Package directory is two levels up: cbf_python/cbf_python
    # Repository directory is three levels up: cbf_python
    curr = Path(__file__).resolve()
    # Find directory containing setup.py or config folder
    for parent in [curr.parent.parent.parent, curr.parent.parent]:
        if (parent / "setup.py").exists() or (parent / "config").exists():
            return parent
    return curr.parent.parent.parent


def get_config_path(config_name: str) -> Path:
    """Resolve full path to a configuration file in config/ directory."""
    root = get_package_root()
    p = root / "config" / config_name
    if not p.suffix:
        p = p.with_suffix(".yaml")
    return p


def resolve_path(relative_or_absolute_path: Union[str, Path]) -> Path:
    """Resolve a relative data or config path against the package root if not absolute."""
    p = Path(relative_or_absolute_path)
    if p.is_absolute():
        return p
    root = get_package_root()
    # Check directly from root
    candidate = root / p
    if candidate.exists():
        return candidate
    # Check in cbf_python subfolder (for data directories)
    candidate_sub = root / "cbf_python" / p
    if candidate_sub.exists():
        return candidate_sub
    return candidate


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionary `override` into `base`."""
    merged: Dict[str, Any] = {}
    for k, v in base.items():
        if isinstance(v, dict):
            merged[k] = v.copy()
        elif isinstance(v, list):
            merged[k] = list(v)
        else:
            merged[k] = v

    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge_dicts(merged[k], v)
        elif isinstance(v, list):
            merged[k] = list(v)
        else:
            merged[k] = v

    return merged


def load_yaml(
    config_path_or_name: Union[str, Path],
    section: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a YAML configuration file as a Python dictionary.

    If `section` is specified, returns only that top-level key's dictionary.
    If the specified file is not found, also checks if a unified `run.yaml` exists
    containing the file's stem as a section (e.g. 'run_cbf_optimal.yaml' -> 'cbf_optimal').
    """
    p = Path(config_path_or_name)
    if not p.exists() or not p.is_file():
        p = get_config_path(str(config_path_or_name))

    if not p.exists():
        run_yaml_path = get_config_path("run.yaml")
        if run_yaml_path.exists():
            stem = Path(config_path_or_name).stem
            candidate_keys = [stem, stem.removeprefix("run_")]
            with open(run_yaml_path, "r", encoding="utf-8") as f:
                run_data = yaml.safe_load(f) or {}
            common_cfg = run_data.get("common", {})
            for k in candidate_keys:
                if k in run_data:
                    return deep_merge_dicts(common_cfg, run_data[k])
        raise FileNotFoundError(f"Configuration file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if section is not None:
        if section not in data:
            raise KeyError(f"Section '{section}' not found in configuration file {p}")
        return data[section]

    return data


def load_run_config(experiment_name: str, config_file: str = "run.yaml") -> Dict[str, Any]:
    """Load configuration for a specific experiment from the unified run.yaml file.

    Merges the 'common' shared parameters with the experiment-specific section.
    """
    sec_key = experiment_name.removeprefix("run_")
    full_data = load_yaml(config_file)
    common_cfg = full_data.get("common", {})
    specific_cfg = full_data.get(sec_key, {})
    if not specific_cfg and sec_key not in full_data and not common_cfg:
        raise KeyError(f"Neither 'common' nor section '{sec_key}' found in {config_file}")
    return deep_merge_dicts(common_cfg, specific_cfg)


def populate_controller_config(
    cfg: ControllerConfig,
    data: Dict[str, Any],
) -> ControllerConfig:
    """Populate a ControllerConfig (or subclass) from a dictionary loaded from YAML."""
    if "Tc" in data:
        cfg.Tc = float(data["Tc"])
    if "C" in data:
        cfg.C = float(data["C"])
    if "Tr" in data:
        cfg.Tr = float(data["Tr"])
    if "a_s" in data:
        cfg.a_s = float(data["a_s"])
    if "gamma" in data:
        cfg.gamma = float(data["gamma"])
    if "max_obstacles" in data:
        cfg.max_obstacles = int(data["max_obstacles"])
    if "lambda_pos" in data:
        cfg.lambda_pos = float(data["lambda_pos"])
    if "lambda_vel" in data:
        cfg.lambda_vel = float(data["lambda_vel"])
    if "lambda_scaling" in data:
        cfg.lambda_scaling = float(data["lambda_scaling"])
    if "lambda_acc" in data:
        cfg.lambda_acc = float(data["lambda_acc"])
    if "DDtrajectory_time_max" in data:
        cfg.DDtrajectory_time_max = float(data["DDtrajectory_time_max"])

    if "prefix" in data:
        cfg.prefix = str(data["prefix"])
    if "tool_frame" in data:
        cfg.tool_frame = str(data["tool_frame"])
    if "elbow_frame" in data:
        cfg.elbow_frame = str(data["elbow_frame"])

    if "Dq_max" in data:
        cfg.Dq_max = np.array(data["Dq_max"], dtype=np.float64)
    if "DDq_max" in data:
        cfg.DDq_max = np.array(data["DDq_max"], dtype=np.float64)
    if "delta_q_max" in data:
        cfg.delta_q_max = np.array(data["delta_q_max"], dtype=np.float64)
    if "delta_unfeasible" in data:
        cfg.delta_unfeasible = np.array(data["delta_unfeasible"], dtype=np.float64)

    # If subclass is PolynomialControllerConfig
    if isinstance(cfg, PolynomialControllerConfig):
        for key in [
            "lambda_0_pos", "lambda_0_vel", "lambda_0_acc", "lambda_0_scaling", "gamma_0",
            "lambda_f_pos", "lambda_f_vel", "lambda_f_acc", "lambda_f_scaling", "gamma_f",
            "n_pos", "n_vel", "n_acc", "n_scaling", "n_gamma",
            "m_pos", "m_vel", "m_acc", "m_scaling", "m_gamma",
            "w_pos", "w_vel", "w_acc", "w_scaling", "w_gamma",
            "h_t",
        ]:
            if key in data:
                setattr(cfg, key, float(data[key]))
        cfg.generate_poly_dict()

    # If subclass is GaussianControllerConfig
    if isinstance(cfg, GaussianControllerConfig):
        if "gaussian_sets" in data:
            cfg.gaussian_sets = []
            for g_data in data["gaussian_sets"]:
                gs = GaussianSet(
                    means=g_data.get("means", {"h": 0.0, "d": 0.0, "v_rel": 0.0}),
                    covariance=np.array(g_data.get("covariance", np.eye(3) * 0.01), dtype=float),
                    lambda_ref=g_data.get("lambda_ref", {
                        "pos": cfg.lambda_pos,
                        "vel": cfg.lambda_vel,
                        "acc": cfg.lambda_acc,
                        "scaling": cfg.lambda_scaling,
                        "gamma": cfg.gamma,
                    }),
                )
                cfg.gaussian_sets.append(gs)
            cfg.n_gaussian_sets = len(cfg.gaussian_sets)
            cfg.precompute_gaussian_parameters()

    return cfg

# Convenient aliases
load_yaml_config = load_yaml
populate_dataclass_from_dict = populate_controller_config
