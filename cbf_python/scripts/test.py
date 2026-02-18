# Script di pulizia per marcare come FAIL i trial vecchi bloccati
import optuna
from optuna.trial import TrialState
from datetime import datetime, timedelta
POSTGRES_URL = "postgresql+psycopg2://optuna:optuna_pw@localhost:5432/optuna_db"

study = optuna.load_study(study_name="...", storage=POSTGRES_URL)

print("Controllo trial bloccati...")
for trial in study.trials:
    # Se è Running da più di 10 minuti, probabilmente è uno zombie
    if trial.state == TrialState.RUNNING:
        # Controllo rozzo basato sull'orario di inizio (se disponibile)
        if trial.datetime_start and (datetime.now() - trial.datetime_start) > timedelta(minutes=10):
            print(f"Marcando trial {trial.number} come FAIL (Zombie)")
            study.tell(trial.number, state=TrialState.FAIL)