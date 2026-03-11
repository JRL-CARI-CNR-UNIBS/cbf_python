import ast

from Controller.Numba_scripts.ssm_cbf_acc import *
import pinocchio as pin
from scripts.util.test_utils import generate_velocity
from Controller.gaussian_controller import GaussianControllerConfig, GaussianSet, GaussianController

import pandas as pd
import numpy as np

from datetime import datetime
import os
Tc: float = 2e-3
C: float = 0.25
Tr: float = 0.5
a_s: float = 4.5

def generate_d_value(h_ref, v_ref):
    d = 0.0
    h = -10
    while h < h_ref:
        h,_ =  h_and_jacobian_numba(d, -0.1, v_ref, 0, Tr, a_s, C, 1e-9)
        d+=0.01
    return d

def compute_required_d(h, v_r, v_h, a_h):
    # 1. Call the function with d = 0.0 to find the base minimum distance
    base_dmin, _ = dmin_and_jacobian_numba(0.0, v_r, v_h, a_h, Tr, a_s, 1e-9)

    # 2. Compute the required d analytically
    required_d = h + C - base_dmin

    return required_d

def generate_pos_sphere(d, ee_x, ee_y, ee_z, model, data, ee_vel): #q, dq, ddq, frame_id):

    v_lin = ee_vel
    v_norm = np.linalg.norm(v_lin)

    # 1. NULL VECTOR CASE: Keep the previous random generation (lower hemisphere)
    if v_norm < 1e-6:
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1.0, 0.0)  # Forces z < ee_z
        sintheta = np.sqrt(1.0 - costheta ** 2)

        x = ee_x + d * sintheta * np.cos(phi)
        y = ee_y + d * sintheta * np.sin(phi)
        z = ee_z + d * costheta
        return np.array([[x, y, z]]).reshape(1,3)

    # 2. NON-NULL VECTOR CASE: Lower half of the 45-degree cone
    z_vec = v_lin / v_norm  # Z-axis of the cone is the reference direction

    # Build a local frame where the local X-axis always points "downwards"
    global_up = np.array([0.0, 0.0, 1.0])
    y_vec = np.cross(global_up, z_vec)  # Horizontal vector perpendicular to the cone

    # Handle the edge case where v_ref is perfectly vertical
    if np.linalg.norm(y_vec) < 1e-6:
        y_vec = np.array([0.0, 1.0, 0.0])
        x_vec = np.array([1.0, 0.0, 0.0])
    else:
        y_vec = y_vec / np.linalg.norm(y_vec)
        # Crossing Y (horizontal) with Z (cone axis) gives an X axis pointing downwards
        x_vec = np.cross(y_vec, z_vec)

        # Rotation matrix to align the local frame to the global frame
    R = np.column_stack((x_vec, y_vec, z_vec))

    # 45 degrees total width implies a half-angle of 22.5 degrees
    cos_half_angle = np.cos(np.deg2rad(22.5))

    # Generate the point on the cone
    # By limiting phi between -pi/2 and pi/2, we force the local X coordinate
    # to be positive, naturally restricting generation to the lower half of the cone
    phi = np.random.uniform(-np.pi / 2, np.pi / 2)
    costheta = np.random.uniform(cos_half_angle, 1.0)
    sintheta = np.sqrt(1.0 - costheta ** 2)

    p_local = np.array([
        d * sintheta * np.cos(phi),
        d * sintheta * np.sin(phi),
        d * costheta
    ])

    # Rotate and translate the local point to global coordinates
    p_rot = R @ p_local
    x = p_rot[0] + ee_x
    y = p_rot[1] + ee_y
    z = p_rot[2] + ee_z

    return np.array([[x, y, z]]).reshape(1,3)

def generate_pos_v_dir(d, ee_x, ee_y, ee_z, v_r):
    """
    Genera un punto a distanza 'd' dal centro (ee_x, ee_y, ee_z)
    nella direzione indicata dal vettore velocità 'v_r'.

    Ritorna:
        np.array di shape (1, 3)
    """
    # 1. Definisci il punto di partenza e il vettore velocità come array
    center = np.array([ee_x, ee_y, ee_z])
    v_r = np.array(v_r)

    # 2. Calcola la norma (lunghezza) del vettore velocità
    norm_v = np.linalg.norm(v_r)

    # 3. Gestisci il caso limite: se la velocità è zero, non c'è direzione.
    # Usiamo una tolleranza piccola (1e-6) per evitare divisioni per zero.
    if norm_v < 1e-6:
        # Puoi decidere come gestire questo caso. Qui ritorniamo la posizione
        # originale, ma potresti voler sollevare un'eccezione o usare un vettore di default.
        return center.reshape(1, 3)

    # 4. Normalizza il vettore (lunghezza = 1) e moltiplicalo per la distanza d
    direction = v_r / norm_v
    offset = direction * d

    # 5. Calcola il nuovo punto
    new_point = center + offset

    # 6. Ritorna il punto come np.array di shape (1, 3)
    return new_point.reshape(1, 3)

def generate_obs_state_h_fixed(obstacle_positions, obstacle_velocities, cycles, enable_spawm, model, data, tool_frame_id, end_eff_pos, Dtrajectory_time, count_move, d_objective, v_ref, spawn_freq,ee_vel):#, q, dq, ddq):
    if (cycles % spawn_freq == 0) and enable_spawm:
        twist = pin.getFrameVelocity(model, data, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        frames_v = twist.linear

        # obstacle_positions = generate_pos_v_dir(d_objective, end_eff_pos[0], end_eff_pos[1], end_eff_pos[2], frames_v)
        obstacle_positions = generate_pos_sphere(d_objective, end_eff_pos[0], end_eff_pos[1], end_eff_pos[2], model.copy(), data.copy(), ee_vel)#, q, dq, ddq, tool_frame_id)
        obstacle_velocities = generate_velocity(end_eff_pos, obstacle_positions, v_ref)
        obstacle_velocities = np.array(obstacle_velocities)
        obstacle_velocities = obstacle_velocities.reshape(1, 3)

    if Dtrajectory_time < 0.05:
        obstacle_positions[0][0] += 0.0015
        obstacle_positions[0][1] += 0.0015
        obstacle_positions[0][2] -= 0.0015
        enable_spawm = False
        count_move += 1
    else:
        enable_spawm = True

    return obstacle_positions, obstacle_velocities, enable_spawm, count_move

def generate_target_h(h_mean, h_std):

    # 2. Sample h from the Gaussian bell curve
    # loc = mean, scale = standard deviation
    h_sampled = np.random.normal(loc=h_mean, scale=h_std)

    return h_sampled

def import_optuna_csv(file_path, study_name):
    """
    Imports Optuna study data from CSV and converts the
    covariance matrix string into a NumPy array.
    """
    # 1. Load the CSV
    df = pd.read_csv(file_path)

    df = df[df['study_name'] == study_name]
    # 2. Convert timestamp to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 3. Safely parse the covariance matrix string into a list/array
    def parse_matrix(matrix_str):
        try:
            # ast.literal_eval is safer than eval() for string-to-list conversion
            return np.array(ast.literal_eval(matrix_str))
        except (ValueError, SyntaxError):
            return None

    if 'user_attrs_covariance_matrix' in df.columns:
        df['user_attrs_covariance_matrix'] = df['user_attrs_covariance_matrix'].apply(parse_matrix)

    return df

def read_config_data_from_csv(cfg: GaussianControllerConfig, filename: str = "../../log_best_trials.csv", study_name = ""):

    df = import_optuna_csv(filename, study_name)
    cfg.lambda_pos = df.iloc[0]['params_lambda_pos']
    cfg.lambda_vel = df.iloc[0]['params_lambda_vel']
    cfg.lambda_acc = df.iloc[0]['params_lambda_acc']
    cfg.lambda_scaling = df.iloc[0]['params_lambda_scaling']
    cfg.gamma = df.iloc[0]['params_gamma']

    if "Gaussian" in type(cfg).__name__:
        gs = GaussianSet()
        gs.lambda_ref = {
            "pos": df.iloc[0]['params_lambda_pos'],
            "vel": df.iloc[0]['params_lambda_vel'],
            "acc": df.iloc[0]['params_lambda_acc'],
            "scaling":  df.iloc[0]['params_lambda_scaling'],
            "gamma": df.iloc[0]['params_gamma']
        }
        gs.covariance = df.iloc[0]['user_attrs_covariance_matrix']
        gs.means = {
            "h": df.iloc[0]['user_attrs_h_mean'],
            "d": df.iloc[0]['user_attrs_d_mean'],
            "v_rel": df.iloc[0]['user_attrs_v_rel_mean']
        }
        cfg.gaussian_sets.append(gs)
        cfg.n_gaussian_sets = len(cfg.gaussian_sets)


def save_data_multiobj(study, filename="log_best_trials.csv"):
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
    weight_v_rate = 0.5  # Penalize violations a bit more heavily
    weight_m_scale = 2.0
    weight_m_err = 1.0
    weight_l_count = 0.3

    df_success["calculated_cost"] = (
            (weight_m_scale * norm_m_scale) +
            (weight_v_rate * norm_v_rate) +
            (weight_m_err * norm_m_err) +
            (weight_l_count * norm_l_count)
    )

    # ... Proceed with your existing sorting and saving logic ...
    top_5 = df_success.sort_values(by="calculated_cost", ascending=False).head(5).copy()
    # ...
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