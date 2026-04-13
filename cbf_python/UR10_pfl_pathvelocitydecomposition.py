import time
import math
import numpy as np
import pinocchio as pin
import meshcat.geometry as mgeom
import quadprog
import matplotlib.pyplot as plt

# classi di utility
from PFLSafetyUtils_Class import PFLSafetyUtils
from pinocchio.visualize import MeshcatVisualizer
from visualization_daemon import VisualizationDaemon
from sharework import loadSharework

from interpolator import SegmentedSE3Trap, SegmentedSE3MinJerk
from pinocchio import SE3
from VisualizationClass import LogPlotter

from QPSolver import QPSolver

# CONFIGURATION
USE_BRIDGE = False  # Set to True to use the real robot bridge, False for fake data from CSV  

# Optimal Parameters
gamma_param = 3.0
k_s_param = 20.0
d_safe_param = 0.1
omega_n_param = 170.0
xi_param = 0.7
w_delta_param = 500.0
w_dds_param = 100.0

# Limits
Dq_MAX = np.pi * np.array([1,1,1,1,1,1], dtype=np.float64) * np.pi
DDq_MAX = np.pi**2*5

# MAIN
def main():
    safety_utilis = PFLSafetyUtils(Tr=0.15, a_s=2.5, v_pfl=0.25, v_max=2.0, rho=20.0, traj_max_err=0.04)
    Tc = 2e-3
    a_s = safety_utilis.a_s
    v_pfl = safety_utilis.v_pfl
    eps_track = safety_utilis.traj_max_err
    v_max = safety_utilis.v_max

    # Model Setup
    UR10E_JOINTS = [
        "ur10e_shoulder_pan_joint", "ur10e_shoulder_lift_joint", "ur10e_elbow_joint",
        "ur10e_wrist_1_joint", "ur10e_wrist_2_joint", "ur10e_wrist_3_joint",
    ]
    model_wrapper = loadSharework(UR10E_JOINTS)
    model = model_wrapper.model
    
    # Visualization Setup
    viz = MeshcatVisualizer(model, model_wrapper.collision_model, model_wrapper.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    
    #visualizzazione ostacoli
    for i in range(20):
        viz.viewer[f"obstacle_{i}"].set_object(mgeom.Sphere(0.1), mgeom.MeshLambertMaterial(color=0xFF0000))
    viz.viewer["goal"].set_object(mgeom.Box([0.2, 0.2, 0.02]), mgeom.MeshLambertMaterial(color=0x00FF00))

    renderer = VisualizationDaemon(viz)

    # Bridge Setup
    target_name = "ur10e_wrist_3_joint"
    home = np.array([90.0, -140.0, 140.0, -90.0, 90.0, 0.0]) * np.pi / 180.0

    if USE_BRIDGE:
        from joint_command_bridge_modified import JointStateCommandBridge
        bridge = JointStateCommandBridge(ordered_joint_names=UR10E_JOINTS, threshold=1.1)
        first_joint_position = bridge.wait_for_first_state(target_name, timeout=5.0)
        if math.isnan(first_joint_position): bridge.shutdown(); return
        first_joint_position = bridge.getPositions()
        bridge.switch_to_forward_position_controller_service()
    else:
        from fake_command_bridge import FakeCommandBridge
        quat = pin.Quaternion(0.814, 0.178, 0.535, 0.137)
        quat.normalize()
        R = quat.toRotationMatrix()
        T_wc = pin.SE3(R, np.array([0.094, -0.93, 2.309]))

        
        csv_path = "/home/nyquist/projects/tesisti/agnelli/cbf_python/skeletons_csv/skeleton_agnelli_1.csv"
        #csv_path = "/home/nyquist/projects/tesisti/agnelli/cbf_python/skeletons_csv/skeleton_vectors_14_NORMAL_TEST1.csv"
        bridge = FakeCommandBridge(UR10E_JOINTS, csv_path=csv_path, Tworld_to_cam=T_wc, slowdown_factor=1.0, t0=0.0)
        first_joint_position = home

    data = model.createData()
    q = first_joint_position.copy()
    dq = np.zeros(model.nq)
    ddq = np.zeros(model.nq) 

    # Frame identification
    tool_frame_id = model.getFrameId("ur10e_tool0") if model.existFrame("ur10e_tool0") else model.getFrameId("tool0")
    pin.framesForwardKinematics(model, data, q)

    print("Control Mode: Path-Velocity Decomposition QP")
    
    # Guadagni smorzati per un comportamento collaborativo
    wn = omega_n_param
    xi = xi_param
    Kp_tra = np.array([1, 1, 1]) * wn ** 2
    Kd_tra = np.array([1, 1, 1]) * 2.0 * xi * wn
    Kp_rot = np.array([1, 1, 1]) * wn ** 2
    Kd_rot = np.array([1, 1, 1]) * 2.0 * xi * wn
    
    planner_cart = SegmentedSE3Trap(vlin_max=1.5, vang_max=0.6, alin_max=0.6, aang_max=0.8)
    
    
    q_start = first_joint_position.copy()
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
    renderer.publishPath(planner_cart.publishPath())

    # Initialization QPclass
    qp_solver = QPSolver(model.nq, Tc, DDq_MAX, Dq_MAX, eps_track, w_delta=w_delta_param, w_dds=w_dds_param)
    qp_solver.enable_velocity_limits = True
    qp_solver.enable_delta_dynamics = True
    qp_solver.enable_scaling_dynamics = True

    # Control Loop Variables
    log_time, log_ds_time = [], []
    log_h, log_scaling = [], []
    log_pos_act, log_pos_nom = [], []
    log_dist, log_vrel = [], []
    log_ddq, log_ddq_nom = [], []
    log_delta = []

    t = 0.0
    trajectory_time = 0.0
    Dtrajectory_time = 1.0
    DDtrajectory_time = 0.0
    h_prev = 100.0
    delta_prev = 0.0
    
    current_dist_min = 100.0

    
    last_obs_pos = []
    last_obs_vel = []
    
    print(f"Starting Simulation. Duration: 50s.")

    try:
        while t < 50.0:
            loop_start = time.perf_counter()

            ### NUOVO: Lettura dal sensore protetta da try-except
            
            if USE_BRIDGE:
                obs_pos, obs_vel, obs_acc = bridge.getObstacles()
            else:
                obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed=t)

            if len(obs_pos) == 0:
                # add a dummy obstacle far away to avoid empty lists
                obs_pos = [np.array([1.0, 1.0, 1.0])]
                obs_vel = [np.zeros(3)]
                obs_acc = [np.zeros(3)]
            
            
            

            # Lettura Task Nominale
            goal_pose, nom_twist, nom_d_twist = planner_cart.getMotionLaw(trajectory_time % T_total)
            pos_nominal = goal_pose.translation.copy()
            
            twist_goal = nom_twist * Dtrajectory_time
            goal_dtwist_base = nom_d_twist * Dtrajectory_time ** 2.0 

            # Cinematica
            pin.framesForwardKinematics(model, data, q)
            pin.computeForwardKinematicsDerivatives(model, data, q, dq, np.zeros(model.nq))
            
            Tbt = data.oMf[tool_frame_id]
            x_curr = Tbt.translation
            Rbt = Tbt.rotation.copy()

            #saturazione errore di posizione
            error_pos = goal_pose.translation - x_curr
            max_pos_err = 0.04  
            err_norm = np.linalg.norm(error_pos)
            if err_norm > max_pos_err:
                error_pos = error_pos * (max_pos_err / err_norm)
                
            error_rot = Rbt @ pin.log3(Rbt.T @ goal_pose.rotation)
            max_rot_err = 0.2  
            err_rot_norm = np.linalg.norm(error_rot)
            if err_rot_norm > max_rot_err:
                error_rot = error_rot * (max_rot_err / err_rot_norm)


            twist_curr = pin.getFrameVelocity(model, data, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                
            # Calcolo Accelerazione Base PD
            acc_lin = Kp_tra * error_pos + Kd_tra * (twist_goal[:3] - twist_curr.linear) + goal_dtwist_base[:3]
            acc_ang = Kp_rot * error_rot + Kd_rot * (twist_goal[3:] - twist_curr.angular) + goal_dtwist_base[3:]
            dtwist_base = np.hstack([acc_lin, acc_ang])
                
            J = pin.computeFrameJacobian(model, data, q, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            dJ = pin.frameJacobianTimeVariation(model, data, q, dq, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jlin = J[:3, :]
            dJlin = dJ[:3, :]

            ks = k_s_param
            d_safe = d_safe_param
            s_dot_target = 1.0 / (1.0 + np.exp(-100 * (current_dist_min - d_safe)))
            s_ddot_des = ks * (s_dot_target - Dtrajectory_time)

            #Qp solver setup
            qp_solver.reset_constraints()
            qp_solver.set_cost_function(J, dJ, dq, dtwist_base, nom_twist, s_ddot_des)

            # Tracking constraints
            next_traj_time_base = trajectory_time + Dtrajectory_time * Tc
            next_pose_des_base, _, _ = planner_cart.getMotionLaw(next_traj_time_base % T_total)
            next_x_des_base = next_pose_des_base.translation
            v_nom_lin = nom_twist[:3].reshape(3, 1)

            qp_solver.add_tracking_and_state_constraints(
                Jlin, dJlin, dq, x_curr, next_x_des_base, v_nom_lin, delta_prev, Dtrajectory_time
            )

            h_min_curr = 100.0
            distance_vector, vrel_vector = [], []
                
            # Ciclo Ostacoli
            for i in range(len(obs_pos)):
                p_o = obs_pos[i]
                v_o = obs_vel[i]
                r = x_curr - p_o
                dist = max(np.linalg.norm(r), 1e-6)
                u_hr = r / dist
                v_rel = np.dot(twist_curr.linear - v_o, u_hr)
                vr_act = np.dot(twist_curr.linear, u_hr)
                
                distance_vector.append(dist)
                vrel_vector.append(v_rel)

                h_softmax_val, dh_softmax_dx = safety_utilis.compute_h_softmax_and_grad(dist, v_rel)
                h_vmax_val, dh_vmax_dx = safety_utilis.compute_h_vmax_and_grad(vr_act)
                
                if h_softmax_val < h_min_curr: h_min_curr = h_softmax_val
                
                f_st, g_st = safety_utilis.range_state_derivative(twist_curr.linear, v_o)
                Jpsi_chi = safety_utilis.jacobian_psi(x_curr, p_o, twist_curr.linear, v_o)
                
                
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
                    
            h_prev = h_min_curr
            dist_min_index = np.argmin(distance_vector)
            current_dist_min = distance_vector[dist_min_index]
            current_vrel_at_min = vrel_vector[dist_min_index]

            # Nominal Acceleration for logging
            ddq_nom = safety_utilis.damped_pinv_svd(J, lam=2e-3) @ (dtwist_base + nom_twist * s_ddot_des - dJ @ dq)

            # === RISOLUZIONE QP ===
            ddq, delta, DDtrajectory_time, success = qp_solver.solve(fallback_dq=dq)
            if not success:
                print(f"QP infeasible (h={h_prev:.2f}) – applying fallback damping.")
                ddq = -10.0 * dq
            delta_prev = delta

            # Integration
            q += dq * Tc + 0.5 * ddq * Tc**2
            dq += ddq * Tc

            dq.clip(-Dq_MAX, Dq_MAX, out=dq) # Aggiornato con il limite corretto invece di [-1, 1]
            ddq.clip(-DDq_MAX, DDq_MAX, out=ddq)
                
            trajectory_time += Dtrajectory_time * Tc + 0.5 * DDtrajectory_time * Tc ** 2.0
            Dtrajectory_time += DDtrajectory_time * Tc
            Dtrajectory_time = np.clip(Dtrajectory_time, 0, 1)

            # Send Command
            if USE_BRIDGE:
                bridge.sendCommand(q)

            # --- LOGGING ---
            log_time.append(t)
            log_ds_time.append(Dtrajectory_time)
            log_h.append(h_prev)
            log_scaling.append(Dtrajectory_time)
            log_pos_act.append(x_curr.copy())
            log_pos_nom.append(pos_nominal)
            log_dist.append(current_dist_min)
            log_vrel.append(current_vrel_at_min)
            log_ddq.append(ddq.copy())
            log_ddq_nom.append(ddq_nom.copy())
            log_delta.append(delta)

            # Viz
            viz_str = f"h={h_prev:.2f} | s_dot={Dtrajectory_time:.2f}"
            renderer.push_state(q, goal_pose, obs_pos, viz_str)
            
            t += Tc
            elapsed = time.perf_counter() - loop_start
            if (Tc - elapsed) > 0:
                time.sleep(Tc - elapsed)

    except KeyboardInterrupt:
        print("Interrupted by User.")
    
    finally:
        if len(log_time) > 0:
            print("\nSimulation ended. Generating Plots...")
            logs = {
                'time': log_time, 'pos_act': log_pos_act, 'pos_nom': log_pos_nom,
                'dist': log_dist, 'vrel': log_vrel, 'h': log_h,
                'ddq': log_ddq, 'ddq_nom': log_ddq_nom,
                'ds_time': log_ds_time, 'delta': log_delta
            }
            config = {
                'v_pfl': v_pfl, 'a_s': a_s, 'Tr': Tc, 'v_max': safety_utilis.v_max
            }
            plotter = LogPlotter(logs, config)
            plotter.show_all_plots()
if __name__ == "__main__":
    main()