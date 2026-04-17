import os
import pandas as pd
from datetime import datetime
import optuna
# from scripts.util.gaussian_process_util import  save_data_multiobj


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
    weight_v_rate = 2.5# Penalize violations a bit more heavily
    weight_m_scale = 2.0
    weight_m_err = 1.5
    weight_l_count = 0.3

    df_success["calculated_cost"] = (
            (weight_m_scale * norm_m_scale) +
            (weight_v_rate * norm_v_rate) +
            (weight_m_err * norm_m_err) +
            (weight_l_count * norm_l_count)
    )

    # ... Proceed with your existing sorting and saving logic ...
    top_5 = df_success.sort_values(by="calculated_cost", ascending=False).head(n_samples).copy()
    # ...
    # 4. Aggiungi timestamp e nome studio per tracciabilità
    top_5.insert(0, 'timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    top_5.insert(1, 'study_name', study.study_name+"revised_case")

    # 5. Seleziona le colonne da salvare (dinamicamente)
    cols_to_keep = (
            ['timestamp', 'study_name', 'number', 'calculated_cost'] +
            [c for c in top_5.columns if c.startswith('values_')] +  # I tuoi 3 obiettivi
            [c for c in top_5.columns if c.startswith('params_')] +  # I parametri
            [c for c in top_5.columns if c.startswith('user_attrs_')]  # Attributi (matrici incluse)
    )
    top_5_clean = top_5[cols_to_keep]

    # 6. SALVATAGGIO INTELLIGENTE
    # Controlla se il file esiste
    file_exists = os.path.isfile(filename)

    # Scrivi in append. Se il file NON esiste, scrivi l'header. Se esiste, no.
    top_5_clean.to_csv(filename, mode='a', header=not file_exists, index=False)

    action = "Creato nuovo file" if not file_exists else "Aggiornato file esistente"
    print(f"{action}: {filename} con i 5 migliori record.")
    
# Define the connection string
storage_url = "postgresql+psycopg2://optuna:optuna_pw@localhost:5432/optuna_db"
study_name = input("Params_name: ")#"params_GPR_test_20260305-165027"

# Load the study
study = optuna.load_study(
    study_name=study_name,
    storage=storage_url
)

# Verify by printing the best parameters found so far
print(f"Study {study_name} loaded successfully.")

save_data_multiobj(study , filename="Dynamic_parameters_results.csv")