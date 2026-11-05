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
    load_controller_params_from_csv,
    populate_controller_from_params_config,
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
        "use_soft_tube": True,
        "lambda_tube": 50000.0,
    }
    populate_controller_config(cfg, data)
    assert cfg.Tc == 0.005
    assert cfg.gamma == 8.0
    assert cfg.lambda_pos == 1234.5
    assert cfg.use_soft_tube is True
    assert cfg.lambda_tube == 50000.0


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


def test_load_controller_params_from_csv():
    """Verify loading optimal controller parameters from CSV with various selectors."""
    cfg = ControllerConfig()

    # 1. By Rank (1 = best)
    meta = load_controller_params_from_csv(cfg, "params_csv/best_10_trials.csv", rank=1, verbose=False)
    assert meta["rank"] == 1
    assert meta["trial_number"] == 2528
    assert abs(cfg.gamma - 2.38196) < 1e-4
    assert abs(cfg.lambda_pos - 67337.3) < 1.0

    # 2. By Rank (2 = 2nd best)
    meta2 = load_controller_params_from_csv(cfg, "params_csv/best_10_trials.csv", rank=2, verbose=False)
    assert meta2["rank"] == 2
    assert meta2["trial_number"] == 4001
    assert abs(cfg.gamma - 2.1979) < 1e-4

    # 3. By Trial Number
    meta3 = load_controller_params_from_csv(cfg, "params_csv/best_10_trials.csv", trial_number=4001, verbose=False)
    assert meta3["trial_number"] == 4001
    assert meta3["rank"] == 2

    # 4. By set_id matching trial number
    meta4 = load_controller_params_from_csv(cfg, "params_csv/best_10_trials.csv", set_id="2528", verbose=False)
    assert meta4["trial_number"] == 2528

    # 5. Out of bounds rank
    with pytest.raises(IndexError):
        load_controller_params_from_csv(cfg, "params_csv/best_10_trials.csv", rank=999, verbose=False)

    # 6. Non-existent trial number
    with pytest.raises(ValueError):
        load_controller_params_from_csv(cfg, "params_csv/best_10_trials.csv", trial_number=999999, verbose=False)


def test_populate_controller_from_params_config():
    """Verify populating ControllerConfig from a parameters dictionary."""
    cfg = ControllerConfig(gamma=99.0)

    # Disabled loading
    res_disabled = populate_controller_from_params_config(
        cfg, {"load_from_csv": False, "params_filename": "params_csv/best_10_trials.csv"}, verbose=False
    )
    assert res_disabled is None
    assert cfg.gamma == 99.0

    # Enabled loading with rank selector
    res_enabled = populate_controller_from_params_config(
        cfg, {"load_from_csv": True, "params_filename": "params_csv/best_10_trials.csv", "rank": 1}, verbose=False
    )
    assert res_enabled is not None
    assert res_enabled["trial_number"] == 2528
    assert abs(cfg.gamma - 2.38196) < 1e-4

