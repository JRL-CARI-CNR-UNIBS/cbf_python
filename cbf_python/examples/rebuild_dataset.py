"""Utility to query Optuna studies from database and rebuild prioritized CSV datasets."""

import itertools
import os
from typing import List, Optional

import optuna
import pandas as pd
from sqlalchemy import create_engine, text

from cbf_python.utils.config_loader import load_yaml
from cbf_python.utils.optimization_helpers import save_data_multiobj


def rebuild_from_prefix(
    storage_url: str,
    prefix: str,
    output_csv: str = "optimal_parameters.csv",
    n_samples: int = 10,
    weights: Optional[List[float]] = None,
) -> None:
    if weights is None:
        weights = [1.5, 2.0, 1.0]

    engine = create_engine(storage_url)
    query = text("SELECT study_name FROM studies WHERE study_name LIKE :prefix")

    with engine.connect() as connection:
        result = connection.execute(query, {"prefix": prefix})
        study_names = [row[0] for row in result]

    if not study_names:
        print(f"No studies found matching prefix '{prefix}'.")
        return

    print(f"Found {len(study_names)} studies matching prefix '{prefix}'.")
    for s_name in sorted(study_names):
        try:
            study = optuna.load_study(study_name=s_name, storage=storage_url)
            save_data_multiobj(study=study, filename=output_csv, n_samples=n_samples, weights=weights)
        except Exception as e:
            print(f"Error processing study '{s_name}': {e}")


def main(config_file: str = "optimization.yaml") -> None:
    opt_cfg = load_yaml(config_file)
    db_url = opt_cfg.get("database", {}).get("url", "")
    prefix = f"{opt_cfg.get('study', {}).get('name_prefix', 'params_optimal')}_%"
    rebuild_from_prefix(storage_url=db_url, prefix=prefix)


if __name__ == "__main__":
    main()
