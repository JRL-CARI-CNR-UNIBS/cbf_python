
from scripts.util.gaussian_process_util import save_data_multiobj
import optuna
from sqlalchemy import create_engine, text

storage_url = "postgresql+psycopg2://optuna:optuna_pw@192.168.66.106:5432/optuna_db"
engine = create_engine(storage_url)



prefix = "GPR_Optimization_h_mean_1.0_v_mean_1.4_%"

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



#save_data_multiobj(study)
