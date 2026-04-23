# -*- coding: utf-8 -*-
"""
Script di Ottimizzazione UR10 PFL - Versione Definitiva (NSGA-II)
"""

import optuna
import numpy as np
import pinocchio as pin
import time
import math
from pinocchio import SE3

from sharework import loadSharework
from fake_command_bridge import FakeCommandBridge
from interpolator import SegmentedSE3Trap

from PFLSafetyUtils_Class import PFLSafetyUtils
from QPSolver import QPSolver

# ----------------- CONFIGURAZIONE DATABASE E COSTANTI -----------------
POSTGRES_URL = "postgresql+psycopg2://optuna:optuna_pw@localhost:5432/optuna_db"

Tc = 2e-3       
DDq_MAX = np.pi**2 * 5
Dq_MAX = np.pi * np.array([1,1,1,1,1,1], dtype=np.float64) * np.pi

# Pesi per le funzioni di costo Multi-Obiettivo
BETA_DELTA = 10.0        
OMEGA_VIOL = 10000.0     
OMEGA_FAILS = 50.0       

# Pesi per chiudere i "loopholes" matematici in J2
W_TRACK = 100.0   # Punisce severamente chi non insegue rigidamente il target cartesiano
W_SMOOTH = 1.0    # Punisce accelerazioni fittizie violente (ks troppo alto)

# ----------------- SIMULAZIONE EPISODIO -----------------
def run_simulation(gamma_param, ks_param, d_safe_param, wn_param, xi_param, w_delta_param, w_dds_param):
    
    safety_utils = PFLSafetyUtils(Tr=0.15, a_s=2.5, v_pfl=0.25, v_max=2.0, rho=20.0, traj_max_err=0.1)
    eps_track = safety_utils.traj_max_err
    v_max = safety_utils.v_max

    qp_solver = QPSolver(nq=6, Tc=Tc, DDq_MAX=DDq_MAX, Dq_MAX=Dq_MAX, eps_track=eps_track, 
                         w_delta=w_delta_param, w_dds=w_dds_param)
    qp_solver.enable_velocity_limits = True
    qp_solver.enable_delta_dynamics = True
    qp_solver.enable_scaling_dynamics = True

    UR10E_JOINTS = [
        "ur10e_shoulder_pan_joint", "ur10e_shoulder_lift_joint", "ur10e_elbow_joint",
        "ur10e_wrist_1_joint", "ur10e_wrist_2_joint", "ur10e_wrist_3_joint",
    ]
    model_wrapper = loadSharework(UR10E_JOINTS)
    model = model_wrapper.model
    data = model.createData()
    
    quat = pin.Quaternion(0.814, 0.178, 0.535, 0.137)
    quat.normalize()
    R = quat.toRotationMatrix()
    T_wc = pin.SE3(R, np.array([0.094, -0.93, 2.309]))

    # CSV path esattamente come nel tuo ambiente
    csv_path = "/home/nyquist/projects/tesisti/agnelli/cbf_python/skeletons_csv/skeleton_agnelli_1.csv"
    bridge = FakeCommandBridge(UR10E_JOINTS, csv_path=csv_path, Tworld_to_cam=T_wc, slowdown_factor=1.0, t0=0.0)
    
    home = np.array([90.0, -140.0, 140.0, -90.0, 90.0, 0.0]) * np.pi / 180.0
    q = home.copy()
    dq = np.zeros(model.nq)
    ddq = np.zeros(model.nq)
    
    tool_frame_id = model.getFrameId("ur10e_tool0") if model.existFrame("ur10e_tool0") else model.getFrameId("tool0")
    pin.framesForwardKinematics(model, data, q)
    
    Kp_tra = np.array([1, 1, 1]) * wn_param ** 2
    Kd_tra = np.array([1, 1, 1]) * 2.0 * xi_param * wn_param
    Kp_rot = np.array([1, 1, 1]) * wn_param ** 2
    Kd_rot = np.array([1, 1, 1]) * 2.0 * xi_param * wn_param

    planner_cart = SegmentedSE3Trap(vlin_max=2.5, vang_max=0.8, alin_max=0.8, aang_max=2.0)
    
    q10 = np.array([31.0, -78.0, 115.0, -127.0, 86.0, -32.0]) * np.pi / 180.0
    q20 = np.array([31.0, -83.0, 98.0, -110.0, 86.0, -32.0]) * np.pi / 180.0
    q22 = np.array([40.0, -126.0, 141.0, -100.0, 86.0, 45.0]) * np.pi / 180.0
    q25 = np.array([130.0, -100.0, 125.0, -115.0, 94.0, -20.0]) * np.pi / 180.0
    q30 = np.array([136.0, -60.0, 90.0, -122.0, 90.0, 45.0]) * np.pi / 180.0
    q40 = np.array([134.0, -65.0, 70.0, -90.0, 90.0, 45.0]) * np.pi / 180.0

    configs = {"q": home, "q10": q10, "q20": q20, "q22": q22, "q25": q25, "q30": q30, "q40": q40}
    ordered_configs = ["q", "q10", "q20", "q10", "q22", "q25", "q30", "q40", "q30", "q"]

    for name in ordered_configs:
        pin.framesForwardKinematics(model, data, configs[name])
        T_ee = data.oMf[tool_frame_id].copy()
        planner_cart.addWayPoint(T_ee)
        
    T_total = planner_cart.computeTime()
    
    # Variabili Loop e metriche
    t = 0.0
    trajectory_time = 0.0
    last_traj_time = 0.0  
    Dtrajectory_time = 1.0
    DDtrajectory_time = 0.0
    delta_prev = 0.0
    
    steps = 0
    qp_fails = 0
    min_h_softmax_viol = 0.0
    min_h_vmax_viol = 0.0
    
    # Metriche cumulative per J2
    sum_sq_scale_penalty = 0.0  
    sum_delta = 0.0             
    sum_pos_error = 0.0
    sum_s_ddot = 0.0
    
    max_duration = 30.0 
    max_simulation_steps = int(max_duration / Tc) 
    
    stuck_time = 0.0
    real_start_time = time.time()  
    task_completed = False         
    abort_reason = "Completed"
    
    try:
        while t < max_duration:
            
            if time.time() - real_start_time > 60.0:
                abort_reason = "Timeout_Hardware"
                break
            
            if steps >= max_simulation_steps:
                abort_reason = "Max_Steps_Reached"
                break
            
            if trajectory_time >= T_total:
                task_completed = True
                abort_reason = "Success"
                break
            
            obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed=t)
            if len(obs_pos) == 0:
                obs_pos, obs_vel, obs_acc = [np.array([10.0, 10.0, 10.0])], [np.zeros(3)], [np.zeros(3)]
            
            goal_pose, nom_twist, nom_d_twist = planner_cart.getMotionLaw(trajectory_time % T_total)
            twist_goal = nom_twist * Dtrajectory_time
            goal_dtwist_base = nom_d_twist * Dtrajectory_time ** 2.0 
            
            pin.framesForwardKinematics(model, data, q)
            pin.computeForwardKinematicsDerivatives(model, data, q, dq, np.zeros(model.nq))
            
            Tbt = data.oMf[tool_frame_id]
            x_curr = Tbt.translation
            Rbt = Tbt.rotation.copy()
            twist_curr = pin.getFrameVelocity(model, data, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            
            J = pin.computeFrameJacobian(model, data, q, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            dJ = pin.frameJacobianTimeVariation(model, data, q, dq, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jlin = J[:3, :]; dJlin = dJ[:3, :]
            
            # --- Calcolo Tracking Error Reale (per la J2) ---
            error_pos_unclamped = goal_pose.translation - x_curr
            sum_pos_error += np.linalg.norm(error_pos_unclamped)

            # Saturazione Errori PD per il controllore
            error_pos = error_pos_unclamped.copy()
            err_norm = np.linalg.norm(error_pos)
            if err_norm > 0.04: error_pos = error_pos * (0.04 / err_norm)
                
            error_rot = Rbt @ pin.log3(Rbt.T @ goal_pose.rotation)
            err_rot_norm = np.linalg.norm(error_rot)
            if err_rot_norm > 0.2: error_rot = error_rot * (0.2 / err_rot_norm)

            acc_lin = Kp_tra * error_pos + Kd_tra * (twist_goal[:3] - twist_curr.linear) + goal_dtwist_base[:3]
            acc_ang = Kp_rot * error_rot + Kd_rot * (twist_goal[3:] - twist_curr.angular) + goal_dtwist_base[3:]
            dtwist_base = np.hstack([acc_lin, acc_ang])
            
            dist_min_all = min([np.linalg.norm(x_curr - p) for p in obs_pos]) if len(obs_pos) > 0 else 10.0
            s_dot_target = 1.0 / (1.0 + np.exp(-100.0 * (dist_min_all - d_safe_param)))
            
            s_ddot_des = ks_param * (s_dot_target - Dtrajectory_time)
            sum_s_ddot += s_ddot_des**2 # Metrica penalità scatti
            
            qp_solver.reset_constraints()
            qp_solver.set_cost_function(J, dJ, dq, dtwist_base, nom_twist, s_ddot_des)

            next_traj_time_base = trajectory_time + Dtrajectory_time * Tc
            next_pose_des_base, _, _ = planner_cart.getMotionLaw(next_traj_time_base % T_total)
            v_nom_lin = nom_twist[:3].reshape(3, 1)

            qp_solver.add_tracking_and_state_constraints(
                Jlin, dJlin, dq, x_curr, next_pose_des_base.translation, v_nom_lin, delta_prev, Dtrajectory_time
            )
            
            h_min_curr = 100.0
            for i in range(len(obs_pos)):
                p_o = obs_pos[i]; v_o = obs_vel[i]
                r = x_curr - p_o  
                dist = max(np.linalg.norm(r), 1e-6)
                u_hr = r / dist
                
                v_rel = np.dot(twist_curr.linear - v_o, u_hr)
                vr_act = np.dot(twist_curr.linear, u_hr)
                
                h_softmax_val, dh_softmax_dx = safety_utils.compute_h_softmax_and_grad(dist, v_rel)
                h_vmax_val, dh_vmax_dx = safety_utils.compute_h_vmax_and_grad(vr_act)
                
                if h_softmax_val < h_min_curr: h_min_curr = h_softmax_val
                    
                if h_softmax_val < 0.0 and vr_act < -1e-4:
                    if h_softmax_val < min_h_softmax_viol: min_h_softmax_viol = h_softmax_val
                if h_vmax_val < 0.0:
                    if h_vmax_val < min_h_vmax_viol: min_h_vmax_viol = h_vmax_val
                
                f_st, g_st = safety_utils.range_state_derivative(twist_curr.linear, v_o)
                Jpsi_chi = safety_utils.jacobian_psi(x_curr, p_o, twist_curr.linear, v_o)
                
                Lfh_softmax = dh_softmax_dx @ Jpsi_chi @ f_st
                Lgh_softmax = dh_softmax_dx @ Jpsi_chi @ g_st
                Apfl = np.hstack([(Lgh_softmax @ Jlin).reshape(1, -1), np.zeros((1, 2))])
                Bpfl = (-Lgh_softmax @ dJlin @ dq - Lfh_softmax - gamma_param * h_softmax_val).reshape(1, -1)
                
                Lfh_vmax = dh_vmax_dx @ Jpsi_chi @ f_st
                Lgh_vmax = dh_vmax_dx @ Jpsi_chi @ g_st
                Avmax = np.hstack([(Lgh_vmax @ Jlin).reshape(1, -1), np.zeros((1, 2))])
                Bvmax = (-Lgh_vmax @ dJlin @ dq - Lfh_vmax - gamma_param * h_vmax_val).reshape(1, -1)
                
                qp_solver.add_custom_constraint(Apfl, Bpfl)
                qp_solver.add_custom_constraint(Avmax, Bvmax)

            # --- Controllo di Sicurezza Anti-Crash C++ ---
            if np.any(np.isnan(dq)) or np.any(np.isnan(dtwist_base)):
                abort_reason = "NaN_Inf_Pre_Solver"
                break

            try:
                ddq, delta_val, DDtrajectory_time, success = qp_solver.solve(fallback_dq=dq)
            except Exception as e:
                success = False
                ddq = np.zeros(model.nq)
                delta_val = 0.0
                DDtrajectory_time = 0.0
            
            if not success:
                qp_fails += 1 
                ddq = -10.0 * dq 
                delta_val = 0.0
                DDtrajectory_time = 0.0
                
            delta_prev = delta_val
            sum_sq_scale_penalty += (1.0 - Dtrajectory_time)**2
            sum_delta += delta_val
            
            q += dq * Tc + 0.5 * ddq * Tc**2
            dq += ddq * Tc
            
            if np.any(np.isnan(q)) or np.any(np.isnan(dq)) or np.any(np.isnan(ddq)) or np.any(np.isinf(ddq)):
                abort_reason = "NaN_Inf_Post_Solver"
                break
                
            dq.clip(-Dq_MAX, Dq_MAX, out=dq)
            ddq.clip(-DDq_MAX, DDq_MAX, out=ddq)
            
            t += Tc
            trajectory_time += Dtrajectory_time * Tc + 0.5 * DDtrajectory_time * Tc**2
            Dtrajectory_time = np.clip(Dtrajectory_time + DDtrajectory_time * Tc, 0.0, 1.0)
            
            progress = trajectory_time - last_traj_time
            if progress < (Tc * 0.01): 
                stuck_time += Tc
            else:
                stuck_time = 0.0
                
            last_traj_time = trajectory_time
            
            if stuck_time >= 3.0: 
                abort_reason = "Stuck"
                break

            steps += 1

    except Exception as e:
        abort_reason = f"Crash_Python"

    # ----------------- COSTRUZIONE FUNZIONI DI COSTO -----------------
    
    completion_ratio = min(1.0, trajectory_time / T_total) if T_total > 0 else 0.0
    missed_task_penalty = 1.0 - completion_ratio

    # J1: Makespan
    if task_completed:
        J1_makespan = steps * Tc
    else:
        J1_makespan = max_duration + (missed_task_penalty * 30.0) 
    
    # J2: Qualità, Slack, Tracciamento e Smoothness
    if steps > 0:
        avg_sq_scale_penalty = sum_sq_scale_penalty / steps
        avg_delta = sum_delta / steps
        avg_pos_error = sum_pos_error / steps
        avg_s_ddot = sum_s_ddot / steps
    else:
        avg_sq_scale_penalty = 1.0
        avg_delta = 1.0
        avg_pos_error = 1.0
        avg_s_ddot = 100.0
        
    J2_quality = avg_sq_scale_penalty + (BETA_DELTA * avg_delta) + (W_TRACK * avg_pos_error) + (W_SMOOTH * avg_s_ddot)
    
    if not task_completed:
        J2_quality += (missed_task_penalty * 10.0)

    # J3: Sicurezza
    viol_score = abs(min_h_softmax_viol) + abs(min_h_vmax_viol)
    J3_safety_stability = (OMEGA_VIOL * viol_score) + (OMEGA_FAILS * qp_fails)

    if str(abort_reason).startswith("NaN_Inf") or str(abort_reason).startswith("Crash"):
        J3_safety_stability += 5000.0 
    elif abort_reason == "Stuck":
        J2_quality += 5.0 

    return J1_makespan, J2_quality, J3_safety_stability, completion_ratio, steps, abort_reason

# ----------------- OPTUNA OBJECTIVE -----------------
def objective(trial):
    gamma_param_trial = trial.suggest_float("gamma", 1.0, 50.0)
    ks_param_trial = trial.suggest_float("k_s", 1.0, 50.0)
    d_safe_trial = trial.suggest_float("d_safe", 0.05, 1.0) 
    
    # FORZATURA OMEGA_N: Limite inferiore alzato a 100.0 per garantire rigidità del robot
    wn_trial = trial.suggest_float("omega_n", 100.0, 300.0)
    xi_trial = trial.suggest_float("xi", 0.5, 1.2)
    w_delta_trial = trial.suggest_float("w_delta", 10.0, 5000.0, log=True) 
    w_dds_trial = trial.suggest_float("w_dds", 10.0, 1000.0)
    
    J1, J2, J3, comp_ratio, steps, abort_reason = run_simulation(
        gamma_param_trial, ks_param_trial, d_safe_trial, 
        wn_trial, xi_trial, w_delta_trial, w_dds_trial
    )
    
    # Salvataggio attributi custom nel database per la dashboard Optuna
    trial.set_user_attr("Completion_Ratio", comp_ratio)
    trial.set_user_attr("Real_Steps", steps)
    trial.set_user_attr("Abort_Reason", str(abort_reason))
    
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

    # Utilizzo NSGA-II: Algoritmo standard industriale per Multi-Obiettivo (genera Frontiere ampie e distribuite)
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(population_size=50),
        storage=storage,
        load_if_exists=True,
        study_name=f"PFL.AGNELLI{time.strftime('%Y%m%d-%H%M%S')}",
    )
    
    # Trial di partenza ragionevole
    study.enqueue_trial({
        "gamma": 10.0, "k_s": 5.0, "d_safe": 0.2,
        "omega_n": 150.0, "xi": 0.7, "w_delta": 100.0, "w_dds": 50.0
    })
    
    print("Avvio Ottimizzazione Multi-Obiettivo (NSGA-II)")
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