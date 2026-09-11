"""
Build GPR Training Dataset from Optuna Grid Optimization Studies.

Applies a configurable Lexicographical selection policy to multi-objective
Pareto fronts to select the optimal hyperparameter vector for each (h, v_rel) cell:
  1. Safety Gate: Adaptive quantile filter q(h) = clip(q0 + beta * h, q_min, q_max)
  2. Smoothness Gate: Cartesian TV error cutoff
  3. Performance Pool: Extracts top-tier fast candidates
  4. Infeasibility Decider: Picks the candidate with minimum solver fallbacks
"""

from __future__ import annotations
import argparse
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sqlalchemy import create_engine, text

from cbf_python.utils.config_loader import load_yaml, resolve_path


def parse_study_coords(study_name: str) -> Optional[Tuple[float, float]]:
    """Extract (h, v) floats from study name like 'gpr_slice_h_m0.025_v_p0.40'."""
    pattern = r"h_([mp][0-9\.]+)_v_([mp][0-9\.]+)"
    match = re.search(pattern, study_name)
    if not match:
        return None

    def token_to_float(tok: str) -> float:
        sign = -1.0 if tok.startswith("m") else 1.0
        return sign * float(tok[1:])

    h_val = token_to_float(match.group(1))
    v_val = token_to_float(match.group(2))
    return round(h_val, 4), round(v_val, 3)


def select_lexicographic_trial(
    df: pd.DataFrame,
    h_target: float,
    lex_cfg: Dict[str, Any],
) -> pd.Series:
    """Select the best trial from a multi-objective study using the 4-tier lexicographical criterion."""
    if df.empty:
        raise ValueError("Cannot select from an empty trials DataFrame.")

    # -------------------------------------------------------------
    # TIER 1: Safety Gate (Adaptive Quantile or Threshold)
    # -------------------------------------------------------------
    safe_cfg = lex_cfg.get("safety", {})
    mode = safe_cfg.get("mode", "quantile")
    safe_col = "values_safety_index_min" if "values_safety_index_min" in df.columns else "values_safety_penalty"
    is_safety_index = (safe_col == "values_safety_index_min")

    if mode == "quantile":
        q0 = float(safe_cfg.get("q0", 0.10))
        beta = float(safe_cfg.get("beta", 1.5))
        q_min = float(safe_cfg.get("q_min", 0.05))
        q_max = float(safe_cfg.get("q_max", 0.85))

        # Adaptive quantile expansion: q(h) = q0 + beta * h
        q_h = float(np.clip(q0 + beta * max(0.0, h_target), q_min, q_max))

        if is_safety_index:
            # Safety Index is maximized (higher is safer): take top q_h portion
            safety_cutoff = float(df[safe_col].quantile(1.0 - q_h))
            t1_candidates = df[df[safe_col] >= safety_cutoff].copy()
        else:
            # Safety penalty is minimized (lower is safer): take lowest q_h portion
            safety_cutoff = float(df[safe_col].quantile(q_h))
            t1_candidates = df[df[safe_col] <= safety_cutoff].copy()
    else:
        if is_safety_index:
            min_s = float(safe_cfg.get("min_safety_index", 0.10))
            t1_candidates = df[df[safe_col] >= min_s].copy()
        else:
            eps0 = float(safe_cfg.get("eps0", 0.0))
            eps_beta = float(safe_cfg.get("eps_beta", 0.20))
            eps_h = max(0.0, eps0 + eps_beta * h_target)
            t1_candidates = df[df[safe_col] <= eps_h].copy()

    if t1_candidates.empty:
        # Fallback: keep the safest available trials
        if is_safety_index:
            max_safe = df[safe_col].max()
            t1_candidates = df[df[safe_col] >= max_safe - 1e-4].copy()
        else:
            min_safe = df[safe_col].min()
            t1_candidates = df[df[safe_col] <= min_safe + 1e-4].copy()

    # -------------------------------------------------------------
    # TIER 2: Smoothness / Jerk Gate
    # -------------------------------------------------------------
    smooth_cfg = lex_cfg.get("smoothness", {})
    max_tv = float(smooth_cfg.get("max_tv_cart", 25.0))
    tv_q = float(smooth_cfg.get("quantile", 0.70))

    tv_cutoff = min(max_tv, float(t1_candidates["values_tv_cart"].quantile(tv_q)))
    t2_candidates = t1_candidates[t1_candidates["values_tv_cart"] <= tv_cutoff].copy()
    if t2_candidates.empty:
        t2_candidates = t1_candidates.copy()

    # -------------------------------------------------------------
    # TIER 3: Performance Extraction Pool
    # -------------------------------------------------------------
    perf_cfg = lex_cfg.get("performance", {})
    top_ratio = float(perf_cfg.get("top_ratio", 0.85))

    p_max = float(t2_candidates["values_performance"].max())
    p_min = float(t2_candidates["values_performance"].min())
    if p_max > 0:
        perf_threshold = p_max * top_ratio
    else:
        perf_threshold = p_max - (p_max - p_min) * (1.0 - top_ratio)

    t3_candidates = t2_candidates[t2_candidates["values_performance"] >= perf_threshold].copy()
    if t3_candidates.empty:
        t3_candidates = t2_candidates.copy()

    # -------------------------------------------------------------
    # TIER 4: Infeasibility Decider & Tie-Breaker
    # -------------------------------------------------------------
    infeas_col = "user_attrs_unfeasible_count" if "user_attrs_unfeasible_count" in t3_candidates.columns else "values_unfeasible_count"
    
    # Sort primarily by lowest solver fallbacks, then highest performance, then lowest jerk
    best_row = t3_candidates.sort_values(
        by=[infeas_col, "values_performance", "values_tv_cart"],
        ascending=[True, False, True],
    ).iloc[0]

    return best_row


def filter_grid_studies(
    study_coords_map: Dict[str, Tuple[float, float]],
    grid_cfg: Dict[str, Any],
) -> Dict[str, Tuple[float, float]]:
    """Filter or downsample studies based on grid density parameters."""
    min_h = float(grid_cfg.get("min_h", -10.0))
    max_h = float(grid_cfg.get("max_h", 10.0))
    min_v = float(grid_cfg.get("min_v", -10.0))
    max_v = float(grid_cfg.get("max_v", 10.0))
    step_h = int(grid_cfg.get("subsample_h_step", 1))
    step_v = int(grid_cfg.get("subsample_v_step", 1))

    # Bounding box filter
    in_box = {
        name: (h, v)
        for name, (h, v) in study_coords_map.items()
        if (min_h <= h <= max_h) and (min_v <= v <= max_v)
    }

    if step_h <= 1 and step_v <= 1:
        return in_box

    # Subsample unique coordinates
    unique_h = sorted(list(set(h for h, _ in in_box.values())))
    unique_v = sorted(list(set(v for _, v in in_box.values())))

    selected_h = set(unique_h[::step_h])
    selected_v = set(unique_v[::step_v])

    return {
        name: (h, v)
        for name, (h, v) in in_box.items()
        if (h in selected_h) and (v in selected_v)
    }


def main(config_file: str = "dataset_builder.yaml") -> None:
    parser = argparse.ArgumentParser(description="Build GPR Dataset via Lexicographic Selection")
    parser.add_argument("--config", type=str, default=config_file, help="Configuration YAML file")
    parser.add_argument("--output", type=str, default=None, help="Optional output CSV path override")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    db_cfg = cfg.get("database", {})
    grid_cfg = cfg.get("grid_selection", {})
    lex_cfg = cfg.get("lexicographic", {})
    out_cfg = cfg.get("output", {})

    output_csv = args.output or out_cfg.get("output_csv", "gpr_training_dataset.csv")
    log_transform = bool(out_cfg.get("log_transform_lambdas", True))

    # Setup storage
    if db_cfg.get("use_sqlite", True):
        sqlite_path = db_cfg.get("sqlite_path", "optuna_gpr_grid.db")
        storage_url = f"sqlite:///{sqlite_path}"
    else:
        storage_url = db_cfg.get("url", "")

    prefix = db_cfg.get("study_prefix", "gpr_slice")
    print(f"Connecting to database: {storage_url}")

    # Query studies
    engine = create_engine(storage_url)
    with engine.connect() as conn:
        res = conn.execute(text("SELECT study_name FROM studies WHERE study_name LIKE :pref"), {"pref": f"{prefix}%"})
        study_names = [row[0] for row in res]

    if not study_names:
        print(f"No Optuna studies found with prefix '{prefix}'.")
        return

    print(f"Found {len(study_names)} studies matching prefix '{prefix}'.")

    # Map study names to coordinates
    study_coords: Dict[str, Tuple[float, float]] = {}
    for s_name in study_names:
        coords = parse_study_coords(s_name)
        if coords:
            study_coords[s_name] = coords

    # Apply grid selection / subsampling
    selected_studies = filter_grid_studies(study_coords, grid_cfg)
    print(f"Selected {len(selected_studies)} grid points according to grid density configuration.")

    dataset_rows: List[Dict[str, Any]] = []

    for idx, (s_name, (h_t, v_t)) in enumerate(sorted(selected_studies.items(), key=lambda x: (x[1][0], x[1][1]))):
        try:
            study = optuna.load_study(study_name=s_name, storage=storage_url)
            df = study.trials_dataframe()
            df_success = df[df["state"] == "COMPLETE"].copy()

            if df_success.empty:
                print(f"Warning: Study '{s_name}' has no completed trials. Skipping.")
                continue

            best = select_lexicographic_trial(df_success, h_target=h_t, lex_cfg=lex_cfg)

            row_data: Dict[str, Any] = {
                "study_name": s_name,
                "h_target": h_t,
                "v_target": v_t,
                "lambda_pos": float(best["params_lambda_pos"]),
                "lambda_vel": float(best["params_lambda_vel"]),
                "lambda_acc": float(best["params_lambda_acc"]),
                "lambda_scaling": float(best["params_lambda_scaling"]),
                "gamma": float(best["params_gamma"]),
                "delta_deg": float(best["params_delta_deg"]),
                "metric_performance": float(best["values_performance"]),
                "metric_safety_index_min": float(best.get("values_safety_index_min", best.get("values_safety_penalty", 0.0))),
                "metric_tv_cart": float(best["values_tv_cart"]),
                "metric_unfeasible_count": int(best.get("user_attrs_unfeasible_count", best.get("values_unfeasible_count", 0))),
            }

            if log_transform:
                row_data["log_lambda_pos"] = float(np.log10(best["params_lambda_pos"]))
                row_data["log_lambda_vel"] = float(np.log10(best["params_lambda_vel"]))
                row_data["log_lambda_acc"] = float(np.log10(best["params_lambda_acc"]))
                row_data["log_lambda_scaling"] = float(np.log10(best["params_lambda_scaling"]))
                row_data["log_gamma"] = float(np.log10(best["params_gamma"]))

            dataset_rows.append(row_data)

        except Exception as e:
            print(f"Error processing study '{s_name}': {e}")

    if not dataset_rows:
        print("No rows generated for dataset.")
        return

    out_df = pd.DataFrame(dataset_rows)
    out_path = resolve_path(output_csv)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nSuccessfully built GPR dataset with {len(out_df)} rows saved to: {out_path}")
    cols_to_show = ["lambda_pos", "lambda_scaling", "gamma", "metric_performance", "metric_safety_index_min", "metric_tv_cart", "metric_unfeasible_count"]
    cols_to_show = [c for c in cols_to_show if c in out_df.columns]
    print(out_df[cols_to_show].describe().to_string())


if __name__ == "__main__":
    main()

