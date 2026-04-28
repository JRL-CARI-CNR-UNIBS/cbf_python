
from scripts.util.gaussian_process_util import save_data_multiobj
import optuna
import itertools
from sqlalchemy import create_engine, text

storage_url = "postgresql+psycopg2://optuna:optuna_pw@192.168.66.100:5432/optuna_db"
engine = create_engine(storage_url)

n_samples = 10
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

def rebuild_generic_ds():
    prefix = f"dynamic_params_polynomial_general_case_%"
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

    save_data_multiobj(study=study, filename=f"../dynamics_par_general_top_{n_samples}.csv", n_samples=n_samples)

# rebuild_generic_ds()
rebuild_gpr_ds()