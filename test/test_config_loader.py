"""Unit tests for YAML configuration loader and path resolvers."""

import os
from pathlib import Path
import numpy as np
import pytest

from cbf_python.controllers.base_optimal_controller import ControllerConfig
from cbf_python.controllers.polynomial_controller import PolynomialControllerConfig
from cbf_python.utils.config_loader import (
    load_yaml,
    populate_controller_config,
    resolve_path,
    get_config_path,
)


def test_load_yaml():
    """Verify loading of standard YAML configs."""
    cfg_data = load_yaml("controller_defaults.yaml")
    assert "Tc" in cfg_data
    assert "gamma" in cfg_data
    assert cfg_data["Tc"] == 0.002


def test_load_run_config():
    """Verify loading unified run configs and section extraction."""
    from cbf_python.utils.config_loader import load_run_config

    opt_cfg = load_run_config("cbf_optimal")
    assert "flags" in opt_cfg
    assert "controller" in opt_cfg
    assert opt_cfg["controller"]["Tc"] == 0.002

    pid_cfg = load_run_config("cbf_pid")
    assert "flags" in pid_cfg
    assert pid_cfg["controller"]["Kp_tra"] == 40.0

    poly_cfg = load_run_config("dynamic_polynomial")
    assert "lambda_0_pos" in poly_cfg["controller"]

    # Test fallback resolution for legacy file name
    fallback_cfg = load_yaml("run_cbf_optimal.yaml")
    assert fallback_cfg["controller"]["gamma"] == opt_cfg["controller"]["gamma"]


def test_populate_controller_config():
    """Verify populating ControllerConfig dataclass from dictionary."""
    cfg = ControllerConfig()
    data = {
        "Tc": 0.005,
        "gamma": 8.0,
        "lambda_pos": 1234.5,
    }
    populate_controller_config(cfg, data)
    assert cfg.Tc == 0.005
    assert cfg.gamma == 8.0
    assert cfg.lambda_pos == 1234.5


def test_populate_polynomial_config():
    """Verify populating PolynomialControllerConfig."""
    cfg = PolynomialControllerConfig()
    data = {
        "lambda_0_pos": 500.0,
        "lambda_f_pos": 100.0,
        "m_pos": 3.0,
    }
    populate_controller_config(cfg, data)
    assert cfg.lambda_0_pos == 500.0
    assert cfg.lambda_f_pos == 100.0
    assert cfg.m_pos == 3.0


def test_resolve_path():
    """Verify resolution of relative paths."""
    resolved = resolve_path("config/bridges.yaml")
    assert resolved.exists()
    assert resolved.is_file()
