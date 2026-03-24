#import os 
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

import optuna
import optunahub
import numpy as np
import pinocchio as pin
import quadprog
import math
import time
from pinocchio import SE3
from interpolator import SegmentedSE3Trap

# Import specifici del progetto
from sharework import loadSharework
from fake_command_bridge import FakeCommandBridge

# ----------------- CONFIGURAZIONE DATABASE -----------------
POSTGRES_URL = "postgresql+psycopg2://optuna:optuna_pw@localhost:5432/optuna_db"

# ----------------- CONFIGURAZIONE FISICA/CBF -----------------
Tr_param = 0.15     
as_param = 2.5      
Tc = 2e-3       
DDq_MAX = np.pi**2 * 5
v_max = 2.0
v_pfl = 0.25
rho = 20.0          # Parametro di asprezza per SoftMin/SoftMax
eps_track = 0.05    # Tolleranza spaziale di tracking (5 cm)

# Pesi per le funzioni di costo Multi-Obiettivo
BETA_DELTA = 1.0         # Peso per la media delle variabili di slack in J2
OMEGA_VIOL = 10000.0     # Penalità gravissima per violazione PFL colpevole in J3
OMEGA_FAILS = 100.0      # Penalità per infeasibility del QP in J3

# ----------------- FUNZIONI MATEMATICHE PFL -----------------

def compute_h_softmax_and_grad(d, v_rel, Tr, a_s, v_pfl, rho):
    h_br = d - (-v_rel*Tr + (v_rel**2) / (2.0 * abs(a_s)))
    grad_br = np.array([1.0, Tr - v_rel / a_s, 0.0])
    
    h_pfl = (v_pfl + v_rel) * Tr
    grad_pfl = np.array([0.0, Tr, 0.0])

    max_inner = max(h_br, h_pfl)
    
    # Log-sum-exp trick per evitare overflow
    exp_br = np.exp(rho * (h_br - max_inner))
    exp_pfl = np.exp(rho * (h_pfl - max_inner))
    sum_inner = exp_br + exp_pfl
    
    h_softmax = max_inner + (1.0 / rho) * np.log(sum_inner)
    
    # Pesi Interni
    omega_br = exp_br / sum_inner
    omega_pfl = exp_pfl / sum_inner
    
    grad_hsoftmax = omega_br * grad_br + omega_pfl * grad_pfl
    
    return h_softmax, grad_hsoftmax

def compute_h_vmax_and_grad(v_max_val, vr_act):
    h_vmax_pos = (v_max_val - vr_act) * (v_max_val + vr_act)
    grad_vmax = np.array([0.0, 0.0, -2 * vr_act])
    return h_vmax_pos, grad_vmax

def range_state_derivative(v_lin, v_human):
    zero3 = np.zeros(3)
    f = np.concatenate([v_lin, v_human, zero3, zero3])
    g = np.zeros((12, 3))
    g[6:9] = np.eye(3)
    return f, g

def jacobian_psi(p_r, p_h, v_lin, v_human):
    diff = p_r - p_h
    norm = np.linalg.norm(diff)
    if norm < 1e-9: norm = 1e-9
    u_rh = (diff / norm).reshape(3, 1)
    P = np.eye(3) - u_rh @ u_rh.T
    
    w = v_lin - v_human
    wP_over_d = (w @ P) / norm
    vrP_over_d = (v_lin @ P) / norm
    
    row_d = np.hstack((u_rh.T, -u_rh.T, np.zeros((1, 3)), np.zeros((1, 3))))
    row_vrel = np.hstack((wP_over_d.reshape(1, -1), -wP_over_d.reshape(1, -1), u_rh.T, -u_rh.T))
    row_vract = np.hstack((vrP_over_d.reshape(1, -1), -vrP_over_d.reshape(1, -1), u_rh.T, np.zeros((1, 3))))
    
    return np.vstack((row_d, row_vrel, row_vract))

def compute_ds_scaling(distance, error, d_thresh_trial):
    # Fattore Sicurezza basato su Distanza fisica (Sigmoide)
    slope_d = 100.0
    term_safety = 1.0 / (1.0 + np.exp(-slope_d * (distance - d_thresh_trial)))
    
    # Fattore Errore basato su Super-Gaussiana
    limit_err = eps_track * 1.5  
    n_power = 10.0               
    term_error = np.exp(- (abs(error) / limit_err)**n_power)

    ds = min(term_safety, term_error)
    return ds

# ----------------- SIMULAZIONE EPISODIO -----------------
def run_simulation(gamma_param, ks_param, d_thresh_param, wn_param, xi_param, w_delta_param):
    # Model Setup
    UR10E_JOINTS = [
        "ur10e_shoulder_pan_joint", "ur10e_shoulder_lift_joint", "ur10e_elbow_joint",
        "ur10e_wrist_1_joint", "ur10e_wrist_2_joint", "ur10e_wrist_3_joint",
    ]
    model_wrapper = loadSharework(UR10E_JOINTS)
    model = model_wrapper.model
    data = model.createData()
    
    # Bridge Setup
    quat = pin.Quaternion(0.814, 0.178, 0.535, 0.137)
    quat.normalize()
    R = quat.toRotationMatrix()
    T_wc = pin.SE3(R, np.array([0.208, -0.883, 2.351]))

    # Camera and bridge
    # # Build camera pose from your INITI snippet
    # quat = pin.Quaternion(0.83, 0.185, 0.513, 0.12)
    # quat.normalize()
    # R = quat.toRotationMatrix()

    # T_wc = pin.SE3(R, np.array([0.094, -0.93, 2.309]))

    csv_path = "/home/nyquist/projects/cells_ws/src/zed_skeleton_kinematics/csv_files/skeleton_vectors_14_NORMAL_TEST1.csv"
    
    bridge = FakeCommandBridge(UR10E_JOINTS, csv_path=csv_path, Tworld_to_cam=T_wc, slowdown_factor=1.0, t0=0.0)
    
    # Inizializzazione Robot
    home = np.array([90.0, -140.0, 140.0, -90.0, 90.0, 0.0]) * np.pi / 180.0
    q = home.copy()
    dq = np.zeros(model.nq)
    ddq = np.zeros(model.nq)
    
    tool_frame_id = model.getFrameId("ur10e_tool0") if model.existFrame("ur10e_tool0") else model.getFrameId("tool0")
    
    # ------------- GAINS CONTROLLORE CARTESIANO (Da Optuna) -------------
    Kp_tra = np.array([1, 1, 1]) * wn_param ** 2
    Kd_tra = np.array([1, 1, 1]) * 2.0 * xi_param * wn_param
    Kp_rot = np.array([1, 1, 1]) * wn_param ** 2
    Kd_rot = np.array([1, 1, 1]) * 2.0 * xi_param * wn_param

    # Planner Cartesian
    planner_cart = SegmentedSE3Trap(vlin_max=0.6, vang_max=1.2, alin_max=1.8, aang_max=2.0)
    
    # --- DEFINIZIONE PUNTI TRAIETTORIA ---
    q_start = home.copy()
    q10 = np.array([31.0, -78.0, 115.0, -127.0, 86.0, -32.0]) * np.pi / 180.0
    q20 = np.array([31.0, -83.0, 98.0, -110.0, 86.0, -32.0]) * np.pi / 180.0
    q22 = np.array([40.0, -126.0, 141.0, -100.0, 86.0, 45.0]) * np.pi / 180.0
    q25 = np.array([130.0, -100.0, 125.0, -115.0, 94.0, -20.0]) * np.pi / 180.0
    q30 = np.array([136.0, -60.0, 90.0, -122.0, 90.0, 45.0]) * np.pi / 180.0
    q40 = np.array([134.0, -65.0, 70.0, -90.0, 90.0, 45.0]) * np.pi / 180.0

    configs = {
        "q": q_start, "q10": q10, "q20": q20, "q22": q22, 
        "q25": q25, "q30": q30, "q40": q40,
    }
    ordered_configs = ["q", "q10", "q20", "q10", "q22", "q25", "q30", "q40", "q30", "q"]

    for name in ordered_configs:
        pin.framesForwardKinematics(model, data, configs[name])
        T_ee = data.oMf[tool_frame_id].copy()
        planner_cart.addWayPoint(T_ee)
        
    T_total = planner_cart.computeTime()
    
    # Variabili Loop
    t = 0.0
    trajectory_time = 0.0
    Dtrajectory_time = 1.0
    DDtrajectory_time = 0.0
    distance = 3.0
    
    # Metriche per ottimizzazione J1, J2, J3
    steps = 0
    qp_fails = 0
    min_h_softmax_viol = 0.0
    min_h_vmax_viol = 0.0
    
    sum_sq_scale_penalty = 0.0  
    sum_delta = 0.0             
    
    max_duration = 150.0 # Timeout limite di sicurezza
    
    try:
        while t < max_duration:
            
            # Condizione di uscita ottimale: il robot ha completato la traiettoria nominale
            if trajectory_time >= T_total:
                break
            
            # Perception
            obs_pos, obs_vel, obs_acc = bridge.getObstacles()
            
            # FK & Jacobians
            pin.framesForwardKinematics(model, data, q)
            pin.computeForwardKinematicsDerivatives(model, data, q, dq, ddq)
            
            Tbt = data.oMf[tool_frame_id]
            x_curr = Tbt.translation
            twist_curr = pin.getFrameVelocity(model, data, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            
            J = pin.computeFrameJacobian(model, data, q, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            dJ = pin.frameJacobianTimeVariation(model, data, q, dq, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jlin = J[:3, :]
            dJlin = dJ[:3, :]
            
            # Tracking Error & Time Scaling
            goal_pose_nominal, _, _ = planner_cart.getMotionLaw(trajectory_time % T_total)
            tracking_error = np.linalg.norm(goal_pose_nominal.translation - x_curr)
            
            Ds_target = np.clip(compute_ds_scaling(distance, tracking_error, d_thresh_param), 0.0, 1.0)
            DDtrajectory_time = ks_param * (Ds_target - Dtrajectory_time)
            
            # Trajectory
            goal_pose, nom_twist, nom_d_twist = planner_cart.getMotionLaw(trajectory_time % T_total)
            twist_goal = nom_twist * Dtrajectory_time
            goal_dtwist = (nom_d_twist * Dtrajectory_time**2 + nom_twist * DDtrajectory_time)
            
            # PD Control
            error_rot = Tbt.rotation @ pin.log3(Tbt.rotation.T @ goal_pose.rotation)
            acc_lin = Kp_tra * (goal_pose.translation - x_curr) + Kd_tra * (twist_goal[:3] - twist_curr.linear) + goal_dtwist[:3]
            acc_ang = Kp_rot * error_rot + Kd_rot * (twist_goal[3:] - twist_curr.angular) + goal_dtwist[3:]
            dtwist_des = np.hstack([acc_lin, acc_ang])
            
            # QP Construction (Variabili: [ddq_1 ... ddq_6, delta])
            constraint_matrix = np.empty((0, model.nq + 1))
            constraint_vector = np.empty((0, 1))
            
            h_min_curr = 100.0
            
            for i in range(len(obs_pos)):
                p_o = obs_pos[i]; v_o = obs_vel[i]
                r = x_curr - p_o  # Vettore da ostacolo a robot
                dist = max(np.linalg.norm(r), 1e-6)
                u_hr = r / dist
                
                v_rel = np.dot(twist_curr.linear - v_o, u_hr)
                vr_act = np.dot(twist_curr.linear, u_hr)
                
                distance = dist # Update for the next step's time scaling
                
                # CBF SoftMax & Vmax
                h_softmax_val, dh_softmax_dx = compute_h_softmax_and_grad(dist, v_rel, Tr_param, as_param, v_pfl, rho)
                h_vmax_val, dh_vmax_dx = compute_h_vmax_and_grad(v_max, vr_act)
                
                if h_softmax_val < h_min_curr: 
                    h_min_curr = h_softmax_val
                    
                # --- LOGICA DI PENALIZZAZIONE ---
                # Colpa in avvicinamento: vr_act < 0 significa che il robot va verso l'ostacolo
                if h_softmax_val < 0.0 and vr_act < -1e-4:
                    if h_softmax_val < min_h_softmax_viol:
                        min_h_softmax_viol = h_softmax_val
                        
                # Colpa in fuga: supera v_max
                if h_vmax_val < 0.0:
                    if h_vmax_val < min_h_vmax_viol:
                        min_h_vmax_viol = h_vmax_val
                
                f_st, g_st = range_state_derivative(twist_curr.linear, v_o)
                Jpsi_chi = jacobian_psi(x_curr, p_o, twist_curr.linear, v_o)
                
                # QP Constraints per PFL
                Lfh_softmax = dh_softmax_dx @ Jpsi_chi @ f_st
                Lgh_softmax = dh_softmax_dx @ Jpsi_chi @ g_st
                Apfl = np.hstack([(Lgh_softmax @ Jlin).reshape(1, -1), np.zeros((1, 1))])
                Bpfl = (-Lgh_softmax @ dJlin @ dq - Lfh_softmax - gamma_param * h_softmax_val).reshape(1, -1)
                
                # QP Constraints per Vmax
                Lfh_vmax = dh_vmax_dx @ Jpsi_chi @ f_st
                Lgh_vmax = dh_vmax_dx @ Jpsi_chi @ g_st
                Avmax = np.hstack([(Lgh_vmax @ Jlin).reshape(1, -1), np.zeros((1, 1))])
                Bvmax = (-Lgh_vmax @ dJlin @ dq - Lfh_vmax - gamma_param * h_vmax_val).reshape(1, -1)
                
                constraint_matrix = np.concatenate((constraint_matrix, Apfl, Avmax), axis=0)
                constraint_vector = np.concatenate((constraint_vector, Bpfl, Bvmax), axis=0)

            # ----------------- LIMITI DI ACCELERAZIONE -----------------
            constraint_acc_mat = np.hstack([np.eye(model.nq), np.zeros((model.nq, 1))])
            constraint_acc_vec = np.ones((model.nq, 1)) * DDq_MAX
            
            # ----------------- LIMITI DI TRACKING -----------------
            next_trajectory_time = trajectory_time + Dtrajectory_time * Tc + 0.5 * DDtrajectory_time * Tc ** 2.0
            next_pose_des, _, _ = planner_cart.getMotionLaw(next_trajectory_time % T_total)
            next_x_des = next_pose_des.translation
            
            constraint_track_matrix_up = np.hstack([-Jlin * 0.5 * Tc**2, np.ones((3, 1))])
            constraint_track_matrix_lower = np.hstack([Jlin * 0.5 * Tc**2, np.ones((3, 1))])
            
            v_curr = Jlin @ dq
            x_free = x_curr + v_curr * Tc + 0.5 * (dJlin @ dq) * Tc**2
            upper_constraint_track_vector = (-eps_track + (x_free - next_x_des)).reshape(-1, 1)
            lower_constraint_track_vector = (-eps_track - (x_free - next_x_des)).reshape(-1, 1)

            constraint_matrix = np.concatenate((
                constraint_matrix,
                constraint_acc_mat,        
                -constraint_acc_mat,       
                constraint_track_matrix_up, 
                constraint_track_matrix_lower   
            ), axis=0)
            
            constraint_vector = np.concatenate((
                constraint_vector,
                -constraint_acc_vec,
                -constraint_acc_vec,
                upper_constraint_track_vector,
                lower_constraint_track_vector
            ), axis=0)
            
            # ----------------- VINCOLO SLACK (DELTA >= 0) -----------------
            mat_delta = np.zeros((1, model.nq + 1))
            mat_delta[0, -1] = 1.0
            constraint_matrix = np.concatenate((constraint_matrix, mat_delta), axis=0)
            constraint_vector = np.concatenate((constraint_vector, np.zeros((1, 1))), axis=0)
            
            # ----------------- RISOLUZIONE QP -----------------
            P_acc = J.T @ J + 1e-6 * np.eye(model.nq)
            b_acc = (J.T @ (dtwist_des - dJ @ dq)).flatten()
            
            # Peso Slack (Da Optuna)
            P_delta = np.array([[w_delta_param]])
            b_delta = np.array([0.0])
            
            zeros_tr = np.zeros((model.nq, 1))
            zeros_bl = np.zeros((1, model.nq))
            
            P = np.block([
                [P_acc,    zeros_tr],
                [zeros_bl, P_delta]
            ])
            b = np.concatenate([b_acc, b_delta])
            
            delta_val = 0.0
            try:
                sol = quadprog.solve_qp(P, b, constraint_matrix.T, constraint_vector.flatten(), 0)
                qp_sol = sol[0]
                ddq = qp_sol[:model.nq]
                delta_val = qp_sol[model.nq]
            except ValueError:
                qp_fails += 1
                ddq = -10.0 * dq 
            
            # ----------------- AGGIORNAMENTO METRICHE E INTEGRAZIONE -----------------
            sum_sq_scale_penalty += (1.0 - Dtrajectory_time)**2
            sum_delta += delta_val
            steps += 1
            
            q += dq * Tc + 0.5 * ddq * Tc**2
            dq += ddq * Tc
            
            t += Tc
            trajectory_time += Dtrajectory_time * Tc + 0.5 * DDtrajectory_time * Tc**2
            Dtrajectory_time = np.clip(Dtrajectory_time + DDtrajectory_time * Tc, 0.0, 1.0)
            
    except Exception as e:
        print(f"Simulation crashed: {e}")
        # Ritorna penalità massime assolute in caso di crash matematico/fisico
        return max_duration, 1000.0, 100000.0

    # ----------------- COSTRUZIONE FUNZIONI DI COSTO (J1, J2, J3) -----------------
    
    # J1: Produttività 
    J1_makespan = steps * Tc
    
    # J2: Qualità Scaling (Fluidità) e Deviazione (Slack)
    if steps > 0:
        avg_sq_scale_penalty = sum_sq_scale_penalty / steps
        avg_delta = sum_delta / steps
    else:
        avg_sq_scale_penalty = 1.0
        avg_delta = 100.0
        
    J2_quality = avg_sq_scale_penalty + (BETA_DELTA * avg_delta)
    
    # J3: Sicurezza e Stabilità (Penalizziamo SOLO le violazioni colpevoli del robot)
    viol_score = abs(min_h_softmax_viol) + abs(min_h_vmax_viol)
    J3_safety_stability = (OMEGA_VIOL * viol_score) + (OMEGA_FAILS * qp_fails)

    return J1_makespan, J2_quality, J3_safety_stability

# ----------------- OPTUNA OBJECTIVE -----------------
def objective(trial):
    # Parametri Barriera e Time Scaling
    gamma_param_trial = trial.suggest_float("gamma", 1.0, 50.0)
    ks_param_trial = trial.suggest_float("k_s", 1.0, 20.0)
    d_thresh_trial = trial.suggest_float("d_threshold", 0.1, 0.8) 
    
    # Parametri Controllore PD Cartesiano
    wn_trial = trial.suggest_float("omega_n", 20.0, 500.0)
    xi_trial = trial.suggest_float("xi", 0.5, 1.2)
    
    # Parametri Solutore QP
    w_delta_trial = trial.suggest_float("w_delta", 10.0, 1000.0, log=True) 
    
    J1, J2, J3 = run_simulation(
        gamma_param_trial, ks_param_trial, d_thresh_trial, 
        wn_trial, xi_trial, w_delta_trial
    )
    
    return J1, J2, J3

# ----------------- MAIN EXECUTION -----------------
if __name__ == "__main__":
    
    storage = optuna.storages.RDBStorage(
        url=POSTGRES_URL,
        engine_kwargs={
            "pool_pre_ping": True,
            "pool_size": 40,
            "max_overflow": 20,
        },
        heartbeat_interval=30,
        grace_period=120,
    )

    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
        storage=storage,
        load_if_exists=True,
        study_name=f"PFL.AGNELLI{time.strftime('%Y%m%d-%H%M%S')}",
    )
    
    print("Starting Multi-Objective Optimization (J1: Makespan, J2: Quality/Slack, J3: Safety/Fails)...")
    # study.set_metric_names(["violation_rate", "mean_scaling", "mean_trajectory_error", "low_scale_rate", "lap count"])

    study.optimize(objective, n_trials=3000, show_progress_bar=True, n_jobs=30)
    
    pareto_front = study.best_trials

    print(f"\nNumero di trial sulla frontiera di Pareto: {len(pareto_front)}")

    for trial in pareto_front:
        print(f"Trial ID: {trial.number}")
        print(f"J1 (Makespan): {trial.values[0]:.3f}s")
        print(f"J2 (Quality) : {trial.values[1]:.5f}")
        print(f"J3 (Safety)  : {trial.values[2]:.2f}")
        print(f"Parametri: {trial.params}")
        print("-" * 30)