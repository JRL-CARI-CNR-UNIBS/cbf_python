"""
Configuration loader for YAML parameters and structured dataclasses.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import yaml

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
    """Resolve a relative data or config path against the package root or params directory."""
    p = Path(relative_or_absolute_path)
    if p.is_absolute() and p.exists():
        return p

    root = get_package_root()

    # Search candidates in priority order
    candidates = [
        root / p,
        root / "cbf_python" / p,
        root / "cbf_python_params" / p,
        root / "cbf_python_params" / "params_csv" / p,
        root / "cbf_python" / "params_csv" / p,
        Path.cwd() / p,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (root / p).resolve()


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
    if "C_0" in data:
        cfg.C_0 = float(data["C_0"])
    if "beta" in data:
        cfg.beta = float(data["beta"])
    if "T_f_bar" in data:
        cfg.T_f_bar = float(data["T_f_bar"])
    if "sigma_p" in data:
        cfg.sigma_p = float(data["sigma_p"])
    if "sigma_v" in data:
        cfg.sigma_v = float(data["sigma_v"])
    if "sigma_a" in data:
        cfg.sigma_a = float(data["sigma_a"])
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
    if "use_soft_tube" in data:
        cfg.use_soft_tube = bool(data["use_soft_tube"])
    if "lambda_tube" in data:
        cfg.lambda_tube = float(data["lambda_tube"])

    if "prefix" in data:
        cfg.prefix = str(data["prefix"])
    if "tool_frame" in data:
        cfg.tool_frame = str(data["tool_frame"])
    if "elbow_frame" in data:
        cfg.elbow_frame = str(data["elbow_frame"])

    for k in ["robot_links", "monitored_links", "capsule_links", "links"]:
        if k in data and data[k] is not None:
            cfg.robot_links = list(data[k])
            break

    if "use_capsules" in data:
        cfg.use_capsules = bool(data["use_capsules"])
    if "robot_capsule_radius" in data:
        cfg.robot_capsule_radius = float(data["robot_capsule_radius"])
    if "human_capsule_radius" in data:
        cfg.human_capsule_radius = float(data["human_capsule_radius"])

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


def load_controller_params_from_csv(
    cfg: ControllerConfig,
    csv_file: Union[str, Path],
    rank: Optional[Union[int, str]] = None,
    trial_number: Optional[Union[int, str]] = None,
    set_id: Optional[Union[int, str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Load optimal controller parameters from a CSV file into a ControllerConfig dataclass.

    Row Selection Priority:
      1. trial_number: Match exact integer in 'number' (or 'trial_number') column.
      2. rank: 1-based integer rank (1 = top/best trial, 2 = 2nd best) or string ('best', 'top').
      3. set_id: General identifier:
         - matches trial number in 'number' column
         - matches string ID in 'ID' column
         - 0-based integer row index if within bounds
      4. Default: Rank 1 (row 0), corresponding to the best trial.

    Parameters populated into ControllerConfig (if present in CSV):
      - gamma:           from 'params_gamma' or 'gamma'
      - lambda_pos:      from 'params_lambda_pos' or 'lambda_pos'
      - lambda_vel:      from 'params_lambda_vel' or 'lambda_vel'
      - lambda_scaling:  from 'params_lambda_scaling' or 'lambda_scaling'
      - lambda_acc:      from 'params_lambda_acc' or 'lambda_acc'
      - lambda_tube:     from 'params_lambda_tube' or 'lambda_tube'
      - delta_deg:       from 'params_delta_deg' or 'delta_deg'

    Returns:
      Dict with metadata regarding the matched trial (trial number, rank, utility, etc.).
    """
    resolved_path = resolve_path(csv_file)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Parameters CSV file not found: {csv_file} (resolved to {resolved_path})")

    df = pd.read_csv(resolved_path)
    if df.empty:
        raise ValueError(f"Parameters CSV file is empty: {resolved_path}")

    selected_idx: Optional[int] = None
    selection_reason: str = ""

    # 1. Match by trial_number
    if trial_number is not None and str(trial_number).strip() != "":
        target_num = int(trial_number)
        num_col = None
        for col in ["number", "trial_number", "trial"]:
            if col in df.columns:
                num_col = col
                break
        if num_col is None:
            raise KeyError(f"Trial number column ('number') not found in {resolved_path}")
        matches = df.index[df[num_col] == target_num].tolist()
        if not matches:
            available = df[num_col].tolist()[:10]
            raise ValueError(f"Trial number {target_num} not found in {resolved_path}. Available trials: {available}")
        selected_idx = matches[0]
        selection_reason = f"trial_number={target_num}"

    # 2. Match by rank
    elif rank is not None and str(rank).strip() != "":
        if isinstance(rank, str) and rank.lower() in ("best", "top", "1st"):
            rank_val = 1
        else:
            try:
                rank_val = int(rank)
            except ValueError:
                raise ValueError(f"Invalid rank value: '{rank}'. Must be a 1-based integer or 'best'.")

        if rank_val < 1 or rank_val > len(df):
            raise IndexError(f"Rank {rank_val} is out of range [1, {len(df)}] for {resolved_path}")
        selected_idx = rank_val - 1
        selection_reason = f"rank={rank_val}"

    # 3. Match by set_id
    elif set_id is not None and str(set_id).strip() != "":
        set_id_str = str(set_id).strip()
        if set_id_str.lower() in ("best", "top"):
            selected_idx = 0
            selection_reason = f"set_id='{set_id_str}' (rank 1)"
        elif set_id_str.lower().startswith("rank:"):
            r_val = int(set_id_str.split(":", 1)[1])
            selected_idx = r_val - 1
            selection_reason = f"set_id='{set_id_str}' (rank {r_val})"
        else:
            # Check 'number' column
            matched = False
            for col in ["number", "trial_number", "trial"]:
                if col in df.columns:
                    try:
                        num_target = int(set_id_str)
                        matches = df.index[df[col] == num_target].tolist()
                        if matches:
                            selected_idx = matches[0]
                            selection_reason = f"set_id={set_id_str} (matched {col})"
                            matched = True
                            break
                    except ValueError:
                        pass
            if not matched and "ID" in df.columns:
                matches = df.index[df["ID"].astype(str) == set_id_str].tolist()
                if matches:
                    selected_idx = matches[0]
                    selection_reason = f"set_id='{set_id_str}' (matched ID)"
                    matched = True

            if not matched:
                try:
                    idx_val = int(set_id_str)
                    if 0 <= idx_val < len(df):
                        selected_idx = idx_val
                        selection_reason = f"set_id={idx_val} (row index)"
                        matched = True
                except ValueError:
                    pass

            if not matched:
                raise ValueError(f"Could not resolve set_id '{set_id}' in {resolved_path}")

    # 4. Default: Best trial (rank 1, row 0)
    else:
        selected_idx = 0
        selection_reason = "default (rank 1 / best)"

    row = df.iloc[selected_idx]

    def _get_val(*candidates: str) -> Optional[float]:
        for c in candidates:
            if c in row and pd.notna(row[c]):
                return float(row[c])
        return None

    loaded: Dict[str, float] = {}

    gamma_val = _get_val("params_gamma", "gamma", "gamma_0")
    if gamma_val is not None:
        cfg.gamma = gamma_val
        loaded["gamma"] = gamma_val

    pos_val = _get_val("params_lambda_pos", "lambda_pos", "lambda_0_pos")
    if pos_val is not None:
        cfg.lambda_pos = pos_val
        loaded["lambda_pos"] = pos_val

    vel_val = _get_val("params_lambda_vel", "lambda_vel", "lambda_0_vel")
    if vel_val is not None:
        cfg.lambda_vel = vel_val
        loaded["lambda_vel"] = vel_val

    scale_val = _get_val("params_lambda_scaling", "lambda_scaling", "lambda_0_scaling")
    if scale_val is not None:
        cfg.lambda_scaling = scale_val
        loaded["lambda_scaling"] = scale_val

    acc_val = _get_val("params_lambda_acc", "lambda_acc", "lambda_0_acc")
    if acc_val is not None:
        cfg.lambda_acc = acc_val
        loaded["lambda_acc"] = acc_val

    tube_val = _get_val("params_lambda_tube", "lambda_tube")
    if tube_val is not None:
        cfg.lambda_tube = tube_val
        loaded["lambda_tube"] = tube_val

    delta_deg_val = _get_val("params_delta_deg", "delta_deg", "delta_0_deg")
    if delta_deg_val is not None:
        loaded["delta_deg"] = delta_deg_val

    meta: Dict[str, Any] = {
        "source_file": str(resolved_path),
        "selection_reason": selection_reason,
        "row_index": selected_idx,
        "rank": selected_idx + 1,
        "trial_number": int(row["number"]) if "number" in row and pd.notna(row["number"]) else None,
        "utility_percentage": float(row["utility_percentage"]) if "utility_percentage" in row and pd.notna(row["utility_percentage"]) else None,
        "calculated_cost": float(row["calculated_cost"]) if "calculated_cost" in row and pd.notna(row["calculated_cost"]) else None,
        "loaded_params": loaded,
        "delta_deg": delta_deg_val,
    }

    if verbose:
        trial_str = f"Trial #{meta['trial_number']}" if meta["trial_number"] is not None else f"Row {selected_idx}"
        util_str = f", Utility: {meta['utility_percentage']:.2f}%" if meta["utility_percentage"] is not None else ""
        print(f"[INFO] Loaded optimal controller parameters from '{resolved_path.name}':")
        print(f"       Selection: Rank {meta['rank']} ({trial_str}{util_str}) via {selection_reason}")
        for k, v in loaded.items():
            if abs(v) < 1e-4 or abs(v) >= 1e4:
                print(f"       {k:<16}: {v:.4e}")
            else:
                print(f"       {k:<16}: {v:.4f}")

    return meta


def populate_controller_from_params_config(
    cfg: ControllerConfig,
    params_config: Dict[str, Any],
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Configure ControllerConfig using parameters dictionary from run.yaml.

    Checks 'load_from_csv' and 'params_filename'. If enabled and filename is given,
    delegates to load_controller_params_from_csv using 'rank', 'trial_number', or 'set_id'.
    """
    if not params_config.get("load_from_csv", True):
        return None

    filename = params_config.get("params_filename")
    if not filename:
        return None

    rank = params_config.get("rank")
    trial_number = params_config.get("trial_number")
    set_id = params_config.get("set_id")

    return load_controller_params_from_csv(
        cfg=cfg,
        csv_file=filename,
        rank=rank,
        trial_number=trial_number,
        set_id=set_id,
        verbose=verbose,
    )

