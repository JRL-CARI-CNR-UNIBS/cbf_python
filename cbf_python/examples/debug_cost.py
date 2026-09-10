"""Diagnostic script for experimenting with multi-objective weight combinations on Optuna studies."""

import optuna
import pandas as pd

from cbf_python.utils.config_loader import load_yaml
from cbf_python.utils.optimization_helpers import save_data_multiobj


def main(config_file: str = "optimization.yaml") -> None:
    opt_cfg = load_yaml(config_file)
    db_cfg = opt_cfg.get("database", {})
    sqlite_path = db_cfg.get("sqlite_path", "optuna_study.db")
    storage = f"sqlite:///{sqlite_path}"

    print(f"Loading studies from {storage} to test cost formulation...")
    try:
        summaries = optuna.get_all_study_summaries(storage=storage)
        if not summaries:
            print("No studies found in storage.")
            return
        latest_study_name = summaries[-1].study_name
        study = optuna.load_study(study_name=latest_study_name, storage=storage)
        print(f"Analyzing latest study: '{latest_study_name}' ({len(study.trials)} trials)")

        test_weights = [
            [1.0, 1.0, 1.0],
            [2.5, 2.0, 1.5],
            [5.0, 1.0, 0.5],
        ]
        for w in test_weights:
            print(f"\n--- Testing weights: tv_cart={w[0]}, scale={w[1]}, err={w[2]} ---")
            save_data_multiobj(study, filename="debug_trials_output.csv", n_samples=3, weights=w)
    except Exception as e:
        print(f"Could not load study from storage: {e}")


if __name__ == "__main__":
    main()
