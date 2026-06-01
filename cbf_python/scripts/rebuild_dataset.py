
# from scripts.util.gaussian_process_util import save_data_multiobj
import optuna
import itertools
from sqlalchemy import create_engine, text
import pandas as pd
import os
from datetime import datetime
import numpy as np
storage_url = "postgresql+psycopg2://optuna:optuna_pw@192.168.66.106:5432/optuna_db"
engine = create_engine(storage_url)

n_samples = 10

def save_data_multiobj(study, filename="log_best_trials.csv", n_samples = 5):
    df = study.trials_dataframe()
    df_success = df[df["state"] == "COMPLETE"].copy()

    if df_success.empty or len(df_success) < 2:
        print("Non ci sono abbastanza trial completati per normalizzare e salvare.")
        return

    # 1. Isolate the metrics
    v_rate = df_success["values_violation_rate"]
    m_scale = df_success["values_mean_scaling"]
    m_err = df_success["values_mean_trajectory_error"]
    l_count = df_success["values_lap count"]

    # 2. Min-Max Normalization (Safe against division by zero)
    def normalize(series, minimize=False):
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            return pd.Series(1.0, index=series.index)  # All trials performed exactly the same

        norm = (series - s_min) / (s_max - s_min)
        return 1.0 - norm if minimize else norm

    # 3. Calculate normalized scores (1.0 is best for ALL of them now)
    norm_v_rate = normalize(v_rate, minimize=True)
    norm_m_scale = normalize(m_scale, minimize=False)
    norm_m_err = normalize(m_err, minimize=True)
    norm_l_count = normalize(l_count, minimize=False)

    # 4. Apply weights to the NORMALIZED values
    # Now, a weight of "1" means "these are equally important relative to their own variance"
    weight_v_rate = 5.0 # Penalize violations a bit more heavily
    weight_m_scale = 2.0
    weight_m_err = 1.0
    weight_l_count = 0.3

    df_success["calculated_cost"] = (
            (weight_m_scale * norm_m_scale) +
            (weight_v_rate * norm_v_rate) +
            (weight_m_err * norm_m_err) +
            (weight_l_count * norm_l_count)
    )
    # --- NEW LOGIC: Sort, Drop Duplicates, THEN take top n_samples ---
    df_sorted = df_success.sort_values(by="calculated_cost", ascending=False)

    # Identify columns to check for uniqueness (both metrics and parameters)
    cols_for_uniqueness = [c for c in df_sorted.columns if c.startswith('values_') or c.startswith('params_')]

    # Drop duplicates keeping the first occurrence (which is the highest cost since it's sorted)
    top_samples = df_sorted.drop_duplicates(subset=cols_for_uniqueness, keep='first').head(n_samples).copy()
    # -----------------------------------------------------------------

    # 4. Aggiungi timestamp e nome studio per tracciabilità
    top_samples.insert(0, 'timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    top_samples.insert(1, 'study_name', study.study_name)

    # 5. Seleziona le colonne da salvare (dinamicamente)
    cols_to_keep = (
            ['timestamp', 'study_name', 'number', 'calculated_cost'] +
            [c for c in top_samples.columns if c.startswith('values_')] +  # I tuoi obiettivi
            [c for c in top_samples.columns if c.startswith('params_')] +  # I parametri
            [c for c in top_samples.columns if c.startswith('user_attrs_')]  # Attributi (matrici incluse)
    )
    top_samples_clean = top_samples[cols_to_keep]

    # 6. SALVATAGGIO INTELLIGENTE
    # Controlla se il file esiste
    file_exists = os.path.isfile(filename)

    # Scrivi in append. Se il file NON esiste, scrivi l'header. Se esiste, no.
    top_samples_clean.to_csv(filename, mode='a', header=not file_exists, index=False)

    action = "Creato nuovo file" if not file_exists else "Aggiornato file esistente"
    print(f"{action}: {filename} con i {len(top_samples_clean)} migliori record unici.")


def save_data_multitrial(study, filename="log_best_trials.csv", n_samples=5):
    df = study.trials_dataframe()
    df_success = df[df["state"] == "COMPLETE"].copy()

    if df_success.empty or len(df_success) < 2:
        print("Non ci sono abbastanza trial completati per normalizzare e salvare.")
        return

    # Funzione di normalizzazione orientata al COSTO (0.0 = Ottimo, 1.0 = Pessimo)
    def normalize_to_cost(series, maximize=False):
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            return pd.Series(0.0, index=series.index)  # Se sono tutti uguali, costo 0

        if maximize:
            # Se vogliamo massimizzare (es. lap count), il max ha costo 0
            return (s_max - series) / (s_max - s_min)
        else:
            # Se vogliamo minimizzare (es. error), il min ha costo 0
            return (series - s_min) / (s_max - s_min)

    # Pesi stabiliti
    weights = {
        "viol_rate": 1.0,  # Da minimizzare
        "mean_scale": 2.0,  # Da massimizzare
        "traj_err": 1.5,  # Da minimizzare
        "lap_count": 1.0 # Da massimizzare
    }

    scenarios = ["h_high", "h_low", "sv"]

    # Calcolo dei 3 costi separati
    for sc in scenarios:
        c_viol = normalize_to_cost(df_success[f"user_attrs_{sc}_viol_rate"], maximize=False)
        c_scale = normalize_to_cost(df_success[f"user_attrs_{sc}_mean_scale"], maximize=True)
        c_err = normalize_to_cost(df_success[f"user_attrs_{sc}_traj_err"], maximize=False)
        c_lap = normalize_to_cost(df_success[f"user_attrs_{sc}_lap_count"], maximize=True)

        df_success[f"cost_{sc}"] = (
                (weights["viol_rate"] * c_viol) +
                (weights["mean_scale"] * c_scale) +
                (weights["traj_err"] * c_err) +
                (weights["lap_count"] * c_lap)
        )

    # Calcolo della distanza Euclidea dal punto (0,0,0)
    df_success["calculated_cost"] = np.sqrt(
        df_success["cost_h_high"] ** 2 +
        df_success["cost_h_low"] ** 2 +
        df_success["cost_sv"] ** 2
    )

    # Ordino in modo crescente (distanza minore = migliore)
    df_sorted = df_success.sort_values(by="calculated_cost", ascending=True)

    # Identifico le colonne per l'unicità
    cols_for_uniqueness = [c for c in df_sorted.columns if c.startswith('user_attrs_') or c.startswith('params_')]

    # Rimuovo i duplicati tenendo il primo (che ora è il costo più basso)
    top_samples = df_sorted.drop_duplicates(subset=cols_for_uniqueness, keep='first').head(n_samples).copy()

    # Aggiunta metadati
    top_samples.insert(0, 'timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    top_samples.insert(1, 'study_name', study.study_name)

    # Selezione colonne finali dinamica
    cols_to_keep = (
            ['timestamp', 'study_name', 'number', 'calculated_cost', 'cost_h_high', 'cost_h_low', 'cost_sv'] +
            [c for c in top_samples.columns if c.startswith('user_attrs_')] +
            [c for c in top_samples.columns if c.startswith('params_')]
    )

    # Gestisco il caso in cui ci siano colonne non presenti per evitare errori
    cols_to_keep = [c for c in cols_to_keep if c in top_samples.columns]
    top_samples_clean = top_samples[cols_to_keep]

    # Salvataggio
    file_exists = os.path.isfile(filename)
    top_samples_clean.to_csv(filename, mode='a', header=not file_exists, index=False)

    action = "Creato nuovo file" if not file_exists else "Aggiornato file esistente"
    print(f"{action}: {filename} con i {len(top_samples_clean)} migliori record unici.")
def rebuild_gpr_ds():
    # First value: -0.1 to 1.0 (step 0.05)
    val1_list = [round(-0.1 + i * 0.05, 2) for i in range(23)]  # 23 steps reach 1.0

    # Second value: 0.2 to 1.4 (step 0.3)
    val2_list = [round(0.2 + i * 0.3, 2) for i in range(5)]     # 5 steps reach 1.4

    # Generate all combinations
    combinations = list(itertools.product(val1_list, val2_list))

    # Create the final dictionary
    par_values = {i: list(comb) for i, comb in enumerate(combinations)}

    for key in par_values:
        h_mean = par_values[key][0]
        v_ref = par_values[key][1]
        prefix = f"GPR_Optimization_h_mean_{h_mean}_v_mean_{v_ref}_%"

        # Query only the study_name column from the studies table
        query = text("SELECT study_name FROM studies WHERE study_name LIKE :prefix")

        with engine.connect() as connection:
            result = connection.execute(query, {"prefix": prefix})
            study_names = [row[0] for row in result]

        if study_names:
            print("Found matches:")
            for name in sorted(study_names):
                print(name)
        else:
            print("No matches found.")
        # Load the study
        study_name = study_names[0]
        study = optuna.load_study(
            study_name=study_names[0],
            storage=storage_url
        )

        # Verify by printing the best parameters found so far
        print(f"Study {study_name} loaded successfully.")

        save_data_multiobj(study=study, filename=f"GPR_optimization_results_top_{n_samples}.csv", n_samples=n_samples)

def rebuild_generic_ds(prefix = f"dynamic_params_polynomial_general_case_%"):

    query = text("SELECT study_name FROM studies WHERE study_name LIKE :prefix")

    with engine.connect() as connection:
        result = connection.execute(query, {"prefix": prefix})
        study_names = [row[0] for row in result]

    if study_names:
        print("Found matches:")
        for name in sorted(study_names):
            print(name)
    else:
        print("No matches found.")
    # Load the study
    study_name = study_names[0]
    study = optuna.load_study(
        study_name=study_names[0],
        storage=storage_url
    )

    # Verify by printing the best parameters found so far
    print(f"Study {study_name} loaded successfully.")

    save_data_multitrial(study=study, filename=f"../dynamics_par_multicase_no_jump_top_{n_samples}.csv", n_samples=n_samples)

rebuild_generic_ds(prefix = "dynamic_params_polynomial_multicase_no_jump_%")
# rebuild_gpr_ds()