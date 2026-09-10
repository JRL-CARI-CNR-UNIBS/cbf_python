"""Plots performance comparisons and metric distributions across controllers."""

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cbf_python.utils.config_loader import resolve_path


def plot_comparison(
    csv_file: str = "results/simulation_data_dynamic_params_comparison.csv",
    save_fig: bool = False,
) -> None:
    csv_path = resolve_path(csv_file)
    if not os.path.exists(csv_path):
        print(f"Data file not found: {csv_path}. Please provide a valid CSV dataset.")
        return

    df = pd.read_csv(csv_path)
    metrics = ["on_target_rate", "lap_count", "viol_rate", "mean_scale", "mean_trajectory_error"]
    valid_metrics = [m for m in metrics if m in df.columns]

    if not valid_metrics:
        print(f"None of the expected metrics found in {csv_path}. Columns available: {df.columns.tolist()}")
        return

    n_plots = len(valid_metrics)
    fig, axes = plt.subplots(nrows=(n_plots + 1) // 2, ncols=2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(valid_metrics):
        ax = axes[i]
        sns.lineplot(
            data=df,
            x="h_mean_test" if "h_mean_test" in df.columns else df.columns[0],
            y=metric,
            hue="test_type" if "test_type" in df.columns else None,
            marker="o",
            ax=ax,
        )
        ax.set_title(metric)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_fig:
        out_png = Path(csv_path).with_suffix(".png")
        plt.savefig(out_png, dpi=300)
        print(f"Figure saved to: {out_png}")
    plt.show()


if __name__ == "__main__":
    plot_comparison()
