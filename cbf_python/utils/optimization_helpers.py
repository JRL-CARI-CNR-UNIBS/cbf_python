"""Helpers for Optuna parameter optimization studies and data persistence."""

import ast
from datetime import datetime
from multiprocessing import Process, Queue
from queue import Empty
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cbf_python.controllers.gaussian_controller import GaussianControllerConfig, GaussianSet
from cbf_python.controllers.polynomial_controller import PolynomialControllerConfig


def save_data(study: Any, filename: str = "trials.csv", top_n: int = 5) -> None:
    """Extract top completed trials by value and append them to a CSV log.

    Args:
        study: Optuna study object.
        filename: Destination CSV path.
        top_n: Number of best trials to record.
    """
    df = study.trials_dataframe()
    df_success = df[df["state"] == "COMPLETE"].sort_values(by="value", ascending=False)
    top_samples = df_success.head(top_n).copy()

    if top_samples.empty:
        print("No completed trials found to save.")
        return

    top_samples.insert(0, "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    top_samples.insert(1, "study_name", study.study_name)

    cols_to_keep = (
        ["timestamp", "study_name", "number", "value"]
        + [c for c in top_samples.columns if c.startswith("params_")]
        + [c for c in top_samples.columns if c.startswith("user_attrs_")]
    )
    cols_to_keep = [c for c in cols_to_keep if c in top_samples.columns]
    top_clean = top_samples[cols_to_keep]

    file_exists = os.path.isfile(filename)
    top_clean.to_csv(filename, mode="a", header=not file_exists, index=False)
    print(f"Appended top {len(top_clean)} trials to: {filename}")


def save_data_multiobj(
    study: Any,
    filename: str = "log_best_trials.csv",
    n_samples: int = 5,
    weights: Optional[List[float]] = None,
) -> None:
    """Normalize multi-objective metrics, apply weights, and save best trials to CSV.

    Args:
        study: Multi-objective Optuna study.
        filename: Destination CSV path.
        n_samples: Number of top trials to record.
        weights: Weights for [mean_tv_cartesian (min), mean_scaling (max), mean_trajectory_error (min)].
    """
    if weights is None:
        weights = [1.0, 1.0, 1.0]

    df = study.trials_dataframe()
    df_success = df[(df["state"] == "COMPLETE") & (df["number"] <= 2000)].copy()

    if df_success.empty or len(df_success) < 2:
        print("Not enough completed trials to normalize and save.")
        return

    v_rate = df_success["values_mean_tv_cartesian"] if "values_mean_tv_cartesian" in df_success else df_success.iloc[:, 0]
    m_scale = df_success["values_mean_scaling"] if "values_mean_scaling" in df_success else df_success.iloc[:, 1]
    m_err = df_success["values_mean_trajectory_error"] if "values_mean_trajectory_error" in df_success else df_success.iloc[:, 2]

    def normalize(series: pd.Series, minimize: bool = False) -> pd.Series:
        s_min, s_max = float(series.min()), float(series.max())
        if s_max == s_min:
            return pd.Series(1.0, index=series.index)
        norm = (series - s_min) / (s_max - s_min)
        return 1.0 - norm if minimize else norm

    norm_v_rate = normalize(v_rate, minimize=True)
    norm_m_scale = normalize(m_scale, minimize=False)
    norm_m_err = normalize(m_err, minimize=True)

    df_success["calculated_cost"] = (
        (weights[1] * norm_m_scale) + (weights[0] * norm_v_rate) + (weights[2] * norm_m_err)
    )

    df_sorted = df_success.sort_values(by="calculated_cost", ascending=False)
    cols_for_uniqueness = [c for c in df_sorted.columns if c.startswith("values_") or c.startswith("params_")]

    top_samples = df_sorted.drop_duplicates(subset=cols_for_uniqueness, keep="first").head(n_samples).copy()
    top_samples.insert(0, "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    top_samples.insert(1, "study_name", study.study_name)

    cols_to_keep = (
        ["timestamp", "study_name", "number", "calculated_cost"]
        + [c for c in top_samples.columns if c.startswith("values_")]
        + [c for c in top_samples.columns if c.startswith("params_")]
        + [c for c in top_samples.columns if c.startswith("user_attrs_")]
    )
    cols_to_keep = [c for c in cols_to_keep if c in top_samples.columns]
    top_clean = top_samples[cols_to_keep]

    file_exists = os.path.isfile(filename)
    top_clean.to_csv(filename, mode="a", header=not file_exists, index=False)
    print(f"Saved {len(top_clean)} best multi-objective trials to: {filename}")


def save_data_multitrial(
    study: Any,
    filename: str = "log_best_trials.csv",
    n_samples: int = 5,
    weights: Optional[List[float]] = None,
    scenarios: Optional[List[str]] = None,
) -> None:
    """Aggregate multiple scenario evaluations into a combined Euclidean cost and log best trials."""
    if weights is None:
        weights = [1.0, 2.0, 1.5, 1.0]
    if scenarios is None:
        scenarios = ["h_high", "h_low", "h_025", "h_05"]

    df = study.trials_dataframe()
    df_success = df[df["state"] == "COMPLETE"].copy()

    if df_success.empty or len(df_success) < 2:
        print("Not enough completed trials to normalize and save.")
        return

    def normalize_to_cost(series: pd.Series, maximize: bool = False) -> pd.Series:
        s_min, s_max = float(series.min()), float(series.max())
        if s_max == s_min:
            return pd.Series(0.0, index=series.index)
        if maximize:
            return (s_max - series) / (s_max - s_min)
        return (series - s_min) / (s_max - s_min)

    cost_sum = 0.0
    for sc in scenarios:
        c_viol = normalize_to_cost(df_success[f"user_attrs_{sc}_viol_rate"], maximize=False)
        c_scale = normalize_to_cost(df_success[f"user_attrs_{sc}_mean_scale"], maximize=True)
        c_err = normalize_to_cost(df_success[f"user_attrs_{sc}_traj_err"], maximize=False)
        c_lap = normalize_to_cost(df_success[f"user_attrs_{sc}_lap_count"], maximize=True)

        single_cost = (
            (weights[0] * c_viol) + (weights[1] * c_scale) + (weights[2] * c_err) + (weights[3] * c_lap)
        )
        df_success[f"cost_{sc}"] = single_cost
        cost_sum += single_cost ** 2

    df_success["calculated_cost"] = np.sqrt(cost_sum)
    df_sorted = df_success.sort_values(by="calculated_cost", ascending=True)

    cols_for_uniqueness = [c for c in df_sorted.columns if c.startswith("user_attrs_") or c.startswith("params_")]
    top_samples = df_sorted.drop_duplicates(subset=cols_for_uniqueness, keep="first").head(n_samples).copy()

    top_samples.insert(0, "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    top_samples.insert(1, "study_name", study.study_name)

    cols_to_keep = (
        ["timestamp", "study_name", "number", "calculated_cost"]
        + [c for c in top_samples.columns if c.startswith("cost_")]
        + [c for c in top_samples.columns if c.startswith("user_attrs_")]
        + [c for c in top_samples.columns if c.startswith("params_")]
    )
    cols_to_keep = [c for c in cols_to_keep if c in top_samples.columns]
    top_clean = top_samples[cols_to_keep]

    file_exists = os.path.isfile(filename)
    top_clean.to_csv(filename, mode="a", header=not file_exists, index=False)
    print(f"Saved {len(top_clean)} best multi-scenario trials to: {filename}")


def _run_worker(target_fn: Callable, args: Tuple, kwargs: Dict, q: Queue) -> None:
    try:
        res = target_fn(*args, **kwargs)
        q.put(("ok", res))
    except Exception as e:
        q.put(("err", repr(e)))


def run_episode_with_timeout(target_fn: Callable, *args: Any, timeout: float = 600.0, **kwargs: Any) -> Any:
    """Execute an evaluation episode function in an isolated process with a hard timeout."""
    q: Queue = Queue()
    p = Process(target=_run_worker, args=(target_fn, args, kwargs, q), daemon=True)
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"Episode exceeded timeout of {timeout}s and was terminated")

    try:
        status, payload = q.get_nowait()
    except Empty:
        raise RuntimeError("Worker process terminated prematurely without returning results")

    if status == "ok":
        return payload
    raise RuntimeError(f"Worker process encountered an error: {payload}")


def import_optuna_csv(
    file_path: str,
    h_mean: Optional[float] = None,
    v_mean: Optional[float] = None,
) -> pd.DataFrame:
    """Import Optuna study data from CSV and parse user covariance matrix strings into ndarrays."""
    df = pd.read_csv(file_path)

    if h_mean is not None and v_mean is not None:
        h_str = f"h_mean_{h_mean}"
        v_str = f"v_mean_{v_mean}"
        df = df[df["study_name"].str.contains(h_str) & df["study_name"].str.contains(v_str)]
        if "calculated_cost" in df.columns:
            df = df.sort_values(by="calculated_cost", ascending=False).head(1)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    def parse_matrix(matrix_str: Any) -> Optional[np.ndarray]:
        try:
            return np.array(ast.literal_eval(str(matrix_str)))
        except (ValueError, SyntaxError):
            return None

    if "user_attrs_covariance_matrix" in df.columns:
        df["user_attrs_covariance_matrix"] = df["user_attrs_covariance_matrix"].apply(parse_matrix)

    return df


def read_config_data_from_csv(
    cfg: GaussianControllerConfig,
    filename: str = "log_best_trials.csv",
    h_mean: float = 0.0,
    v_mean: float = 0.0,
) -> None:
    """Populate a GaussianControllerConfig from logged Optuna trials."""
    df = import_optuna_csv(filename, h_mean=h_mean, v_mean=v_mean)
    if df.empty:
        raise ValueError(f"No matching configuration found in {filename} for h_mean={h_mean}, v_mean={v_mean}")

    row = df.iloc[0]
    print(f"Loaded configuration: {row['study_name']}")
    cfg.lambda_pos = float(row["params_lambda_pos"])
    cfg.lambda_vel = float(row["params_lambda_vel"])
    cfg.lambda_acc = float(row["params_lambda_acc"])
    cfg.lambda_scaling = float(row["params_lambda_scaling"])
    cfg.gamma = float(row["params_gamma"])

    gs = GaussianSet()
    gs.lambda_ref = {
        "pos": float(row["params_lambda_pos"]),
        "vel": float(row["params_lambda_vel"]),
        "acc": float(row["params_lambda_acc"]),
        "scaling": float(row["params_lambda_scaling"]),
        "gamma": float(row["params_gamma"]),
    }
    if "user_attrs_covariance_matrix" in row and row["user_attrs_covariance_matrix"] is not None:
        gs.covariance = row["user_attrs_covariance_matrix"]
    if "user_attrs_h_mean" in row:
        gs.means = {
            "h": float(row["user_attrs_h_mean"]),
            "d": float(row["user_attrs_d_mean"]),
            "v_rel": float(row["user_attrs_v_rel_mean"]),
        }
    cfg.gaussian_sets.append(gs)
    cfg.n_gaussian_sets = len(cfg.gaussian_sets)


def read_poly_config_data_from_csv(
    cfg: PolynomialControllerConfig,
    filename: str = "log_best_trials.csv",
    trial_name: str = "",
) -> None:
    """Populate a PolynomialControllerConfig from logged Optuna polynomial trials."""
    df = pd.read_csv(filename)
    if trial_name:
        df = df[df["study_name"].str.contains(trial_name)]
    if df.empty:
        raise ValueError(f"No matching trial for '{trial_name}' in {filename}")

    if "calculated_cost" in df.columns:
        df = df.sort_values(by="calculated_cost", ascending=True)

    row = df.iloc[0]
    cfg.lambda_0_pos = float(row["params_lambda_0_pos"])
    cfg.lambda_0_vel = float(row["params_lambda_0_vel"])
    cfg.lambda_0_acc = float(row["params_lambda_0_acc"])
    cfg.lambda_0_scaling = float(row["params_lambda_0_scaling"])
    cfg.gamma_0 = float(row["params_gamma_0"])

    cfg.lambda_f_pos = float(row["params_lambda_f_pos"])
    cfg.lambda_f_vel = float(row["params_lambda_f_vel"])
    cfg.lambda_f_acc = float(row["params_lambda_f_acc"])
    cfg.lambda_f_scaling = float(row["params_lambda_f_scaling"])
    cfg.gamma_f = float(row["params_gamma_f"])

    cfg.n_pos = float(row["params_n_pos"])
    cfg.n_vel = float(row["params_n_vel"])
    cfg.n_acc = float(row["params_n_acc"])
    cfg.n_scaling = float(row["params_n_scaling"])
    cfg.n_gamma = float(row["params_n_gamma"])

    cfg.m_pos = float(row["params_m_pos"])
    cfg.m_vel = float(row["params_m_vel"])
    cfg.m_acc = float(row["params_m_acc"])
    cfg.m_scaling = float(row["params_m_scaling"])
    cfg.m_gamma = float(row["params_m_gamma"])

    cfg.w_pos = float(row["params_w_pos"])
    cfg.w_vel = float(row["params_w_vel"])
    cfg.w_acc = float(row["params_w_acc"])
    cfg.w_scaling = float(row["params_w_scaling"])
    cfg.w_gamma = float(row["params_w_gamma"])

    cfg.lambda_pos = cfg.lambda_0_pos
    cfg.lambda_vel = cfg.lambda_0_vel
    cfg.lambda_scaling = cfg.lambda_0_scaling
    cfg.lambda_acc = cfg.lambda_0_acc
    cfg.gamma = cfg.gamma_0
