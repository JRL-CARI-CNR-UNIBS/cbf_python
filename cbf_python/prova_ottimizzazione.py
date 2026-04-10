# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 00:55:04 2026

@author: Pietro
"""

import optuna
import optunahub
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
OMEGA_FAILS = 500.0      

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
    sum_sq_scale_penalty = 0.0  
    sum_delta = 0.0             
    
    max_duration = 30.0 
    max_simulation_steps = int(max_duration / Tc) 
    
    stuck_time = 0.0
    real_start_time = time.time()  
    task_completed = False         
    
    try:
        while t < max_duration:
            
            # --- 1. KILL-SWITCH HARDWARE (Se il PC è bloccato) ---
            if time.time() - real_start_time > 300.0:
                print("Timeout hardware superato. CPU in stallo.")
                qp_fails += 1000.0
                break
            
            # --- 2. KILL-SWITCH ITERAZIONI (Hard Cap) ---
            if steps >= max_simulation_steps:
                print("Raggiunto il numero massimo di iterazioni.")
                qp_fails += 500.0
                break
            
            # --- 3. CHECK VITTORIA ---
            if trajectory_time >= T_total:
                task_completed = True
                break
            
            # Perception
            obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed=t)
            if len(obs_pos) == 0:
                obs_pos, obs_vel, obs_acc = [np.array([10.0, 10.0, 10.0])], [np.zeros(3)], [np.zeros(3)]
            
            # Task Nominale
            goal_pose, nom_twist, nom_d_twist = planner_cart.getMotionLaw(trajectory_time % T_total)
            twist_goal = nom_twist * Dtrajectory_time
            goal_dtwist_base = nom_d_twist * Dtrajectory_time ** 2.0 
            
            # FK & Jacobians
            pin.framesForwardKinematics(model, data, q)
            pin.computeForwardKinematicsDerivatives(model, data, q, dq, np.zeros(model.nq))
            
            Tbt = data.oMf[tool_frame_id]
            x_curr = Tbt.translation
            Rbt = Tbt.rotation.copy()
            twist_curr = pin.getFrameVelocity(model, data, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            
            J = pin.computeFrameJacobian(model, data, q, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            dJ = pin.frameJacobianTimeVariation(model, data, q, dq, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jlin = J[:3, :]; dJlin = dJ[:3, :]
            
            # Saturazione Errori PD
            error_pos = goal_pose.translation - x_curr
            err_norm = np.linalg.norm(error_pos)
            if err_norm > 0.04: error_pos = error_pos * (0.04 / err_norm)
                
            error_rot = Rbt @ pin.log3(Rbt.T @ goal_pose.rotation)
            err_rot_norm = np.linalg.norm(error_rot)
            if err_rot_norm > 0.2: error_rot = error_rot * (0.2 / err_rot_norm)

            # Base PD Acceleration
            acc_lin = Kp_tra * error_pos + Kd_tra * (twist_goal[:3] - twist_curr.linear) + goal_dtwist_base[:3]
            acc_ang = Kp_rot * error_rot + Kd_rot * (twist_goal[3:] - twist_curr.angular) + goal_dtwist_base[3:]
            dtwist_base = np.hstack([acc_lin, acc_ang])
            
            # SSM Target
            dist_min_all = min([np.linalg.norm(x_curr - p) for p in obs_pos]) if len(obs_pos) > 0 else 10.0
            s_dot_target = 1.0 / (1.0 + np.exp(-100.0 * (dist_min_all - d_safe_param)))
            s_ddot_des = ks_param * (s_dot_target - Dtrajectory_time)

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

            # Integrazione
            ddq, delta_val, DDtrajectory_time, success = qp_solver.solve(fallback_dq=dq)
            
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
            
            # --- 4. KILL-SWITCH NaN/INF ---
            if np.any(np.isnan(q)) or np.any(np.isnan(dq)) or np.any(np.isnan(ddq)) or np.any(np.isinf(ddq)):
                print("Instabilità numerica (NaN/Inf). Abortisco.")
                qp_fails += 500.0
                break
                
            dq.clip(-Dq_MAX, Dq_MAX, out=dq)
            ddq.clip(-DDq_MAX, DDq_MAX, out=ddq)
            
            t += Tc
            trajectory_time += Dtrajectory_time * Tc + 0.5 * DDtrajectory_time * Tc**2
            Dtrajectory_time = np.clip(Dtrajectory_time + DDtrajectory_time * Tc, 0.0, 1.0)
            
            # --- 5. KILL-SWITCH PROGRESSIONE (Anti-vibrante) ---
            progress = trajectory_time - last_traj_time
            if progress < (Tc * 0.01): 
                stuck_time += Tc
            else:
                stuck_time = 0.0
                
            last_traj_time = trajectory_time
            
            if stuck_time >= 3.0: 
                print("Stallo di progressione (Robot non avanza da 3s). Abortisco.")
                qp_fails += 500.0
                break

            steps += 1

    except Exception as e:
        print(f"Simulation crashed: {e}")
        return 100.0, 1000.0, 100000.0  # Penalità gravissime su tutto

    # ----------------- COSTRUZIONE FUNZIONI DI COSTO (J1, J2, J3) -----------------
    if task_completed:
        J1_makespan = steps * Tc
    else:
        J1_makespan = max_duration + 100.0 
    
    if steps > 0:
        avg_sq_scale_penalty = sum_sq_scale_penalty / steps
        avg_delta = sum_delta / steps
    else:
        avg_sq_scale_penalty = 1.0
        avg_delta = 100.0
        
    J2_quality = avg_sq_scale_penalty + (BETA_DELTA * avg_delta)
    viol_score = abs(min_h_softmax_viol) + abs(min_h_vmax_viol)
    J3_safety_stability = (OMEGA_VIOL * viol_score) + (OMEGA_FAILS * qp_fails)

    return J1_makespan, J2_quality, J3_safety_stability

# ----------------- OPTUNA OBJECTIVE -----------------
def objective(trial):
    gamma_param_trial = trial.suggest_float("gamma", 1.0, 50.0)
    ks_param_trial = trial.suggest_float("k_s", 1.0, 50.0)
    d_safe_trial = trial.suggest_float("d_safe", 0.05, 1.0) 
    wn_trial = trial.suggest_float("omega_n", 20.0, 200.0)
    xi_trial = trial.suggest_float("xi", 0.5, 1.2)
    w_delta_trial = trial.suggest_float("w_delta", 10.0, 5000.0, log=True) 
    w_dds_trial = trial.suggest_float("w_dds", 10.0, 1000.0)
    
    J1, J2, J3 = run_simulation(
        gamma_param_trial, ks_param_trial, d_safe_trial, 
        wn_trial, xi_trial, w_delta_trial, w_dds_trial
    )
    
    return J1, J2, J3

# ----------------- MAIN EXECUTION (DEBUG MODE) -----------------
if __name__ == "__main__":
    print("Avvio singolo trial (Sanity Check)...")
    print("Utilizzo i parametri estratti dallo script di simulazione standalone.")
    
    # Parametri esatti presi da UR10_pfl_pathvelocitydecomposition.py
    J1, J2, J3 = run_simulation(
        gamma_param=10.0, 
        ks_param=5.0, 
        d_safe_param=0.1, 
        wn_param=100.0, 
        xi_param=0.7, 
        w_delta_param=100.0, 
        w_dds_param=1.0
    )
    
    print(f"\n--- RISULTATO FINALE DEL TRIAL ---")
    print(f"J1 (Makespan): {J1:.3f} s")
    print(f"J2 (Quality): {J2:.3f}")
    print(f"J3 (Safety): {J3:.3f}")