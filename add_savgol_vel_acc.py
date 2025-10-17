#!/usr/bin/env python3
"""
add_savgol_vel_acc.py
Compute Savitzky–Golay velocity and acceleration columns for numeric signals.

Usage:
  python add_savgol_vel_acc.py --input a01_s10_e02_skeleton3D_converted.csv \
      --output a01_s10_e02_skeleton3D_with_savgol_vel_acc.csv \
      --time-col time --fps 30 --window 7 --poly 3

Notes:
- If --time-col is provided, dt is inferred from its average positive diff.
- If --fps is provided (and time-col is not), dt=1/fps.
- Otherwise dt=1.0 (unit step).
- Columns to process default to all numeric columns except the time column.
- You can restrict processing with --include and/or --exclude patterns (comma-separated substrings).
"""

import argparse
import sys
from typing import List, Optional
import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
except Exception as e:
    print("scipy is required: pip install scipy", file=sys.stderr)
    raise

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add Savitzky–Golay velocity and acceleration columns")
    p.add_argument("--input", "-i", required=True, help="Input CSV path")
    p.add_argument("--output", "-o", required=True, help="Output CSV path")
    p.add_argument("--time-col", default=None, help="Name of time column (optional)")
    p.add_argument("--fps", type=float, default=None, help="Sampling rate in Hz (used if no time column)")
    p.add_argument("--window", type=int, default=None, help="Savitzky–Golay window length (odd, > poly)")
    p.add_argument("--poly", type=int, default=3, help="Savitzky–Golay polyorder")
    p.add_argument("--include", default=None, help="Comma-separated substrings; only matching columns processed")
    p.add_argument("--exclude", default=None, help="Comma-separated substrings; matching columns skipped")
    p.add_argument("--suffix-vel", default="_vel", help="Velocity column suffix")
    p.add_argument("--suffix-acc", default="_acc", help="Acceleration column suffix")
    return p.parse_args()

def infer_dt(df: pd.DataFrame, time_col: Optional[str], fps: Optional[float]) -> float:
    if time_col and time_col in df.columns:
        dt_series = pd.to_numeric(df[time_col], errors="coerce").diff()
        positive = dt_series[dt_series > 0]
        if positive.size:
            dt = float(positive.mean())
            if np.isfinite(dt) and dt > 0:
                return dt
    if fps and fps > 0:
        return 1.0 / float(fps)
    return 1.0

def choose_window(n: int, poly: int, dt: float, user_window: Optional[int]) -> int:
    if user_window:
        w = int(user_window)
    else:
        # Aim for ~0.2 s window if possible; fallback to ~11 samples
        target = int(round(0.2 / dt)) if dt > 0 else 11
        if target < 5:
            target = 5
        w = target
    # Must be odd
    if w % 2 == 0:
        w += 1
    # Clamp to data length
    w = min(w, n if n % 2 == 1 else n - 1)
    # Ensure > poly
    min_ok = poly + 2 if (poly + 2) % 2 == 1 else poly + 3
    w = max(w, min_ok)
    # Re-clamp
    w = min(w, n if n % 2 == 1 else n - 1)
    if w <= poly:
        # Last-resort: shrink poly
        raise ValueError(f"Window too small (w={w}) for poly={poly}; reduce poly or provide larger window/longer data.")
    return w

def list_numeric_columns(df: pd.DataFrame, time_col: Optional[str], include: Optional[List[str]], exclude: Optional[List[str]]) -> List[str]:
    cols = []
    for c in df.columns:
        if time_col and c == time_col:
            continue
        series = pd.to_numeric(df[c], errors="coerce")
        if series.notna().mean() <= 0.5:
            continue
        name = str(c)
        if include and not any(sub in name for sub in include):
            continue
        if exclude and any(sub in name for sub in exclude):
            continue
        cols.append(c)
    return cols

def savgol_derivative(series: pd.Series, deriv: int, window: int, poly: int, dt: float) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s_interp = s.interpolate(limit_direction="both")
    try:
        out = savgol_filter(s_interp.to_numpy(), window_length=window, polyorder=poly,
                            deriv=deriv, delta=dt, mode="interp")
    except Exception:
        # Fallback to finite differences if S-G fails
        if deriv == 1:
            out = np.gradient(s_interp.to_numpy(), dt)
        else:
            out = np.gradient(np.gradient(s_interp.to_numpy(), dt), dt)
    return out

def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    time_col = args.time_col if (args.time_col in df.columns) else None
    include = [s for s in (args.include.split(",") if args.include else []) if s]
    exclude = [s for s in (args.exclude.split(",") if args.exclude else []) if s]

    dt = infer_dt(df, time_col, args.fps)
    n = len(df)
    window = choose_window(n, args.poly, dt, args.window)

    numeric_cols = list_numeric_columns(df, time_col, include, exclude)
    if not numeric_cols:
        print("No numeric columns selected for processing.", file=sys.stderr)
        sys.exit(2)

    for col in numeric_cols:
        v = savgol_derivative(df[col], deriv=1, window=window, poly=args.poly, dt=dt)
        a = savgol_derivative(df[col], deriv=2, window=window, poly=args.poly, dt=dt)
        df[f"{col}{args.suffix_vel}"] = v
        df[f"{col}{args.suffix_acc}"] = a

    print(f"dataframe = {df}")
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Rows: {len(df)} | Processed columns: {len(numeric_cols)}")
    if time_col:
        print(f"Time column: {time_col} | dt: {dt:.6f} s")
    elif args.fps:
        print(f"fps: {args.fps} | dt: {dt:.6f} s")
    else:
        print(f"Assumed unit dt: {dt:.6f}")

if __name__ == "__main__":
    main()