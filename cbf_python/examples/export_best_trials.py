#!/usr/bin/env python3
"""
Extract, normalize, and rank top Optuna trials from a SQLite/RDB database.

Loads an Optuna multi-objective optimization study, performs 0-1 min-max
normalization across all objective metrics, computes a weighted sum utility score,
and exports the top N trials to a CSV file in the params folder.
"""

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd


# ==============================================================================
# CONFIGURATION CONSTANTS (Modify these as needed)
# ==============================================================================
# Path to SQLite database file or SQLAlchemy connection URL:
DB_PATH: str = "dbs/optimal_controller_opt_result.db"

# Optuna study name to load (set to None to automatically pick the latest study in DB):
STUDY_NAME: Optional[str] = "params_optimal_20261101-001040"

# Output destination CSV file path in the params directory:
OUTPUT_CSV_PATH: str = "params_csv/best_10_trials.csv"

# Number of top trials to extract and save:
TOP_N: int = 10

# Objective weights for the multi-criteria weighted sum ranking:
# Can be specified as a dictionary matching metric names or as a list in objective order:
#   1. Safety penetration (minimize): weight 3.0
#   2. Velocity scaling (maximize):    weight 1.5
#   3. Smoothness / Jerk (minimize):  weight 1.0
#   4. Trajectory error (minimize):   weight 1.0
#   5. Infeasibility count (minimize): weight 3.0
WEIGHTS: Union[Dict[str, float], List[float]] = {
    "safety_penetration": 3.0,
    "scaling": 1.5,
    "smoothness": 1.0,
    "trajectory_error": 1.0,
    "unfeasible_count": 3.0,
}

# Metric keyword aliases to ensure seamless matching across legacy and homogenized metric names
METRIC_ALIASES: Dict[str, List[str]] = {
    "safety_penetration": ["safety", "penetration"],
    "scaling": ["scaling", "scale", "velocity"],
    "smoothness": ["smoothness", "jerk", "oscillation"],
    "trajectory_error": ["trajectory_error", "cartesian_error", "cart_err", "traj_err", "error"],
    "unfeasible_count": ["unfeasible", "infeasible", "fallback"],
}

# Exclude solver fallback / numerical failure penalty trials (e.g. values >= 5.0)
# to prevent artificial penalty spikes from distorting the 0-1 normalization bounds:
FILTER_FALLBACK_TRIALS: bool = True
FALLBACK_PENALTY_THRESHOLD: float = 5.0
# ==============================================================================


def resolve_storage_url(db_path: str) -> str:
    """Ensure db_path is formatted as a valid SQLAlchemy storage URL."""
    if "://" in db_path:
        return db_path
    p = Path(db_path).resolve()
    return f"sqlite:///{p}"


def match_weight(metric_name: str, weights: Union[Dict[str, float], List[float]], idx: int) -> float:
    """Resolve weight for a given metric from dict or list format with alias matching."""
    if isinstance(weights, (list, tuple)):
        return float(weights[idx]) if idx < len(weights) else 1.0

    name_lower = metric_name.lower()

    # 1. Direct key match
    for k, w in weights.items():
        if k.lower() in name_lower or name_lower in k.lower():
            return float(w)

    # 2. Alias group match
    for canonical_key, aliases in METRIC_ALIASES.items():
        if any(alias in name_lower for alias in aliases):
            for k, w in weights.items():
                if k.lower() == canonical_key.lower() or any(alias in k.lower() for alias in aliases):
                    return float(w)

    # 3. Fallback to positional order
    keys = list(weights.keys())
    if idx < len(keys):
        return float(weights[keys[idx]])

    return 1.0


def export_best_trials(
    db_path: str = DB_PATH,
    study_name: Optional[str] = STUDY_NAME,
    output_csv: str = OUTPUT_CSV_PATH,
    top_n: int = TOP_N,
    weights: Union[Dict[str, float], List[float]] = WEIGHTS,
    filter_fallbacks: bool = FILTER_FALLBACK_TRIALS,
    fallback_threshold: float = FALLBACK_PENALTY_THRESHOLD,
) -> pd.DataFrame:
    """Load study, normalize metrics to [0, 1], rank by weighted sum, and save to CSV."""
    storage_url = resolve_storage_url(db_path)
    print(f"Connecting to database: {storage_url}")

    # Auto-detect study name if omitted
    if not study_name:
        summaries = optuna.study.get_all_study_summaries(storage_url)
        if not summaries:
            raise ValueError(f"No studies found in storage: {storage_url}")
        # Select study with highest number of trials
        summaries_sorted = sorted(summaries, key=lambda s: s.n_trials, reverse=True)
        study_name = summaries_sorted[0].study_name
        print(f"Auto-selected study with most trials: '{study_name}' ({summaries_sorted[0].n_trials} trials)")

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    df = study.trials_dataframe()

    if df.empty:
        raise ValueError(f"Study '{study_name}' contains no trials.")

    # 1. Filter completed trials
    df_complete = df[df["state"] == "COMPLETE"].copy()
    print(f"Total trials: {len(df)} | Completed: {len(df_complete)}")

    if df_complete.empty:
        raise ValueError("No trials in COMPLETE state found.")

    # 2. Extract objective columns and directions
    metric_names = getattr(study, "metric_names", None)
    directions = getattr(study, "directions", None)

    obj_cols: List[str] = []
    is_minimize_list: List[bool] = []

    if metric_names:
        for i, name in enumerate(metric_names):
            # Check candidate column names in dataframe
            candidate = f"values_{name}"
            if candidate in df_complete.columns:
                col = candidate
            else:
                matches = [c for c in df_complete.columns if name in c and c.startswith("values_")]
                col = matches[0] if matches else f"values_{i}"

            obj_cols.append(col)
            dir_i = directions[i] if directions and i < len(directions) else optuna.study.StudyDirection.MINIMIZE
            is_min = (dir_i == optuna.study.StudyDirection.MINIMIZE) or ("MIN" in getattr(dir_i, "name", "").upper())
            is_minimize_list.append(is_min)
    else:
        obj_cols = [c for c in df_complete.columns if c.startswith("values_")]
        for i, col in enumerate(obj_cols):
            dir_i = directions[i] if directions and i < len(directions) else optuna.study.StudyDirection.MINIMIZE
            is_min = (dir_i == optuna.study.StudyDirection.MINIMIZE) or ("MIN" in getattr(dir_i, "name", "").upper())
            is_minimize_list.append(is_min)

    print("\nIdentified Optimization Objectives:")
    for i, (col, is_min) in enumerate(zip(obj_cols, is_minimize_list)):
        dir_str = "MINIMIZE (lower is better)" if is_min else "MAXIMIZE (higher is better)"
        w_val = match_weight(col.removeprefix("values_"), weights, i)
        print(f"  [{i + 1}] {col:<32} -> {dir_str:<32} (Weight: {w_val:.2f})")

    # 3. Optional fallback/outlier filtering
    df_valid = df_complete.copy()
    if filter_fallbacks:
        valid_mask = pd.Series(True, index=df_complete.index)
        for col, is_min in zip(obj_cols, is_minimize_list):
            if is_min:
                threshold = 900.0 if ("jerk" in col.lower() or "unfeasible" in col.lower()) else fallback_threshold
                valid_mask &= (df_complete[col] < threshold)
            else:
                valid_mask &= (df_complete[col] > 0.05)
        df_valid = df_complete[valid_mask].copy()
        n_filtered = len(df_complete) - len(df_valid)
        print(f"Filtered {n_filtered} numerical fallback/penalty outliers (threshold: {fallback_threshold}). Remaining valid: {len(df_valid)}")

    if df_valid.empty:
        df_valid = df_complete.copy()

    # 4. Perform 0-1 Normalization (Utility Mapping)
    # 1.0 always represents the best performance; 0.0 represents the worst.
    weighted_score = pd.Series(0.0, index=df_valid.index)
    sum_weights = 0.0

    print("\nObjective Normalization Bounds [0, 1]:")
    for i, (col, is_min) in enumerate(zip(obj_cols, is_minimize_list)):
        s = df_valid[col]
        s_min, s_max = float(s.min()), float(s.max())
        name = col.removeprefix("values_")
        w = match_weight(name, weights, i)
        sum_weights += w

        if s_max > s_min:
            norm_raw = (s - s_min) / (s_max - s_min)
        else:
            norm_raw = pd.Series(1.0, index=s.index)

        # Map to utility score in [0, 1] where 1.0 is best
        norm_utility = (1.0 - norm_raw) if is_min else norm_raw
        df_valid[f"norm_{name}"] = norm_utility
        weighted_score += w * norm_utility

        print(f"  {name:<28} range: [{s_min:10.4g}, {s_max:10.4g}] | Weight: {w:4.1f}")

    df_valid["calculated_cost"] = weighted_score
    df_valid["utility_percentage"] = (weighted_score / max(1e-6, sum_weights)) * 100.0

    # 5. Sort by weighted score descending (best trials on top)
    df_sorted = df_valid.sort_values(by="calculated_cost", ascending=False).copy()
    top_samples = df_sorted.head(top_n).copy()

    # 6. Organize and order columns logically
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_samples.insert(0, "timestamp", now_str)
    top_samples.insert(1, "study_name", study_name)

    param_cols = sorted([c for c in top_samples.columns if c.startswith("params_")])
    user_attr_cols = sorted([c for c in top_samples.columns if c.startswith("user_attrs_")])
    norm_cols = [f"norm_{c.removeprefix('values_')}" for c in obj_cols if f"norm_{c.removeprefix('values_')}" in top_samples.columns]

    ordered_cols = (
        ["timestamp", "study_name", "number", "calculated_cost", "utility_percentage"]
        + norm_cols
        + obj_cols
        + param_cols
        + user_attr_cols
    )
    final_cols = [c for c in ordered_cols if c in top_samples.columns]
    result_df = top_samples[final_cols].copy()

    # 7. Save to CSV
    out_path = Path(output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"\nSuccessfully saved top {len(result_df)} trials to: {out_path}")

    # 8. Print formatted summary table to console
    print("\n" + "=" * 115)
    print(f"TOP {len(result_df)} TRIALS RANKING (Study: {study_name})")
    print("=" * 115)

    display_cols = (
        ["number", "calculated_cost", "utility_percentage"]
        + obj_cols
        + [p for p in param_cols if p in ["params_gamma", "params_lambda_pos", "params_lambda_vel", "params_lambda_scaling"]]
    )
    disp = result_df[display_cols].copy()
    disp.columns = [c.replace("values_", "val_").replace("params_", "p_") for c in disp.columns]
    print(disp.to_string(index=False, justify="right"))
    print("=" * 115)

    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export top Optuna trials to CSV with 0-1 normalization.")
    parser.add_argument("--db", default=DB_PATH, help=f"Path to SQLite DB or URL (default: {DB_PATH})")
    parser.add_argument("--study", default=STUDY_NAME, help="Optuna study name (default: auto/constant)")
    parser.add_argument("--out", default=OUTPUT_CSV_PATH, help=f"Output CSV path (default: {OUTPUT_CSV_PATH})")
    parser.add_argument("--top", type=int, default=TOP_N, help=f"Number of top trials to export (default: {TOP_N})")
    args = parser.parse_args()

    export_best_trials(
        db_path=args.db,
        study_name=args.study,
        output_csv=args.out,
        top_n=args.top,
    )
