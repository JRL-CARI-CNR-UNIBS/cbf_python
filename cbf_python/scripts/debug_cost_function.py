import optuna
from scripts.util.gaussian_process_util import  save_data_multiobj



# Define the connection string
storage_url = "postgresql+psycopg2://optuna:optuna_pw@192.168.66.106:5432/optuna_db"
study_name = input("Params_name: ")#"params_GPR_test_20260305-165027"

# Load the study
study = optuna.load_study(
    study_name=study_name,
    storage=storage_url
)

# Verify by printing the best parameters found so far
print(f"Study {study_name} loaded successfully.")

save_data_multiobj(study , filename="../Dynamic_parameters_results.csv")