import optuna
import pandas as pd
from datetime import datetime
import os

def save_data_multiobj(study, filename="log_best_trials.csv"):
    """
    Salva i top 5 trial ordinati secondo la funzione di costo personalizzata.
    Crea il file se non esiste, altrimenti aggiunge i dati in coda.
    """

    # 1. Recupera i dati
    df = study.trials_dataframe()
    df_success = df[df["state"] == "COMPLETE"].copy()

    if df_success.empty:
        print("Nessun trial completato da salvare.")
        return

    # 2. Calcola la funzione di costo personalizzata
    # Usa i nomi definiti nel tuo set_metric_names
    # Formula: mean_scaling - 10 * violation_rate - mean_trajectory_error
    try:
        df_success["calculated_cost"] = (
                df_success["values_mean_scaling"] -
                df_success["values_violation_rate"] -
                df_success["values_mean_trajectory_error"] +
                df_success["values_lap count"] / 200
        )
    except KeyError:
        print("ERRORE: Colonne non trovate. Verifica i nomi con study.trials_dataframe().columns")
        return

    # 3. Ordina e seleziona i Top 5
    top_5 = df_success.sort_values(by="calculated_cost", ascending=False).head(5).copy()

    # 4. Aggiungi timestamp e nome studio per tracciabilità
    top_5.insert(0, 'timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    top_5.insert(1, 'study_name', study.study_name)

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
storage_url = "postgresql+psycopg2://optuna:optuna_pw@192.168.66.106:5432/optuna_db"
study_name = "params_GPR_test_20260307-103036"

# Load the study
study = optuna.load_study(
    study_name=study_name,
    storage=storage_url
)

# Verify by printing the best parameters found so far
print(f"Study {study_name} loaded successfully.")

save_data_multiobj(study)