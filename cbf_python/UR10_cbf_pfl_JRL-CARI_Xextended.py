import time
import math
import numpy as np
import pinocchio as pin
import meshcat.geometry as mgeom
import quadprog
import matplotlib.pyplot as plt
from PFLSafetyUtils_Class import PFLSafetyUtils

from pinocchio.visualize import MeshcatVisualizer
from visualization_daemon import VisualizationDaemon
from sharework import loadSharework
# from optimal_cbf_task_controller import BCFOptimalController, ControllerConfig

from interpolator import SegmentedSE3Trap, SegmentedSE3MinJerk
from joint_interpolator import SegmentedJointTrap
from pinocchio import SE3
from VisualizationClass import ThesisPlotter

# CONFIGURATION
USE_BRIDGE = False  # Set to True to use the real robot bridge, False for fake data

# QP constraints Parameters
gamma_param =5.0
Dq_MAX = np.pi * np.array([1,1,1,1,1,1], dtype=np.float64) * np.pi
DDq_MAX = np.pi**2*5

# MAIN
def main():

    safety_utilis = PFLSafetyUtils(Tc=0.005, a_s=2.5, v_pfl=0.25, v_max=2.0, rho=20.0, traj_max_err=0.1)
    Tc = safety_utilis.Tc
    a_s = safety_utilis.a_s
    v_pfl = safety_utilis.v_pfl
    eps_track = safety_utilis.traj_max_err
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
    
    # Visualizziamo solo le mani (2 ostacoli)
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

        # T_wc = pin.SE3(R, np.array([0.208, -0.883, 2.351]))
        T_wc = pin.SE3(R, np.array([0.094, -0.93, 2.309]))

        #csv_path = "/home/nyquist/projects/cells_ws/src/zed_skeleton_kinematics/csv_files/skeleton_vectors_22.csv"
        # csv_path = "/home/nyquist/projects/cells_ws/src/zed_skeleton_kinematics/csv_files/skeleton_vectors_14_NORMAL_TEST1.csv"
        csv_path = "/home/nyquist/projects/tesisti/agnelli/cbf_python/skeletons_csv/skeleton_agnelli_1.csv"

        # csv_path = "C:/Users/Pietro/OneDrive/Desktop/cbf_python/skeleton_vectors_22.csv"
        #csv_path = "C:/Users/Pietro/OneDrive/Desktop/cbf_python/UR10_obst.csv"
        bridge = FakeCommandBridge(UR10E_JOINTS, csv_path=csv_path, Tworld_to_cam=T_wc, slowdown_factor=1.0, t0=0.0)
        first_joint_position = home

    # Controller Initialization
    ctrl_prof = None
    planner_joint = None
    planner_cart = None
    
    data = model.createData()
    q = first_joint_position.copy()
    q_des = first_joint_position.copy()
    dq = np.zeros(model.nq)
    ddq = np.zeros(model.nq) # Init acc
    
    delta = 0.0 #Init delta

    # Frame identification
    tool_frame_id = model.getFrameId("ur10e_tool0") if model.existFrame("ur10e_tool0") else model.getFrameId("tool0")
    pin.framesForwardKinematics(model, data, q)
    current_pose_cart = data.oMf[tool_frame_id]


    print("Control Mode: Manual QP")
    wn = 100.0
    xi = 0.3
    Kp_tra = np.array([1, 1, 1]) * wn ** 2
    Kd_tra = np.array([1, 1, 1]) * 2.0 * xi * wn
    Kp_rot = np.array([1, 1, 1]) * wn ** 2
    Kd_rot = np.array([1, 1, 1]) * 2.0 * xi * wn
    
    #planner_cart = SegmentedSE3Trap(vlin_max=0.06, vang_max=0.12, alin_max=1.8, aang_max=2.0)
    
    #planner_cart = SegmentedSE3Trap(vlin_max=0.6, vang_max=0.8, alin_max=1.8, aang_max=2.0)
    planner_cart = SegmentedSE3MinJerk(vlin_max=0.6, vang_max=0.8, alin_max=1.8, aang_max=2.0)
    
    #planner_cart = SegmentedSE3Trap(vlin_max=0.1, vang_max=0.3, alin_max=0.8, aang_max=0.5)
    # planner_cart = SegmentedSE3Trap(vlin_max=2.5, vang_max=3, alin_max=80, aang_max=50)
    q_start = first_joint_position.copy()
    q10 = np.array([31.0, -78.0, 115.0, -127.0, 86.0, -32.0]) * np.pi / 180.0
    q20 = np.array([31.0, -83.0, 98.0, -110.0, 86.0, -32.0]) * np.pi / 180.0
    q22 = np.array([40.0, -126.0, 141.0, -100.0, 86.0, 45.0]) * np.pi / 180.0
    q25 = np.array([130.0, -100.0, 125.0, -115.0, 94.0, -20.0]) * np.pi / 180.0
    q30 = np.array([136.0, -60.0, 90.0, -122.0, 90.0, 45.0]) * np.pi / 180.0
    q40 = np.array([134.0, -65.0, 70.0, -90.0, 90.0, 45.0]) * np.pi / 180.0

    configs = {
        "q": q_start,
        "q10": q10,
        "q20": q20,
        "q22": q22,
        "q25": q25,
        "q30": q30,
        "q40": q40,
    }
    ordered_configs = ["q", "q10", "q20", "q10", "q22", "q25", "q30", "q40", "q30", "q"]

    for name in ordered_configs:
        pin.framesForwardKinematics(model, data, configs[name])
        T_ee = data.oMf[tool_frame_id].copy()
        planner_cart.addWayPoint(T_ee)
        
    T_total = planner_cart.computeTime()
    renderer.publishPath(planner_cart.publishPath())

    # Control Loop Variables
    log_time = []; log_ds_time = [];
    log_h, log_scaling = [], []
    log_pos_act, log_pos_nom = [], []
    log_dist, log_vrel = [], []
    log_ddq , log_ddq_nom = [], []
    log_delta = []

    t = 0.0
    trajectory_time = 0.0
    Dtrajectory_time = 1.0
    DDtrajectory_time = 0.0
    h_prev = 100.0
    h_min = 100.0
    
    v_rel_max = 2.5
    v_pfl = 0.25
   

    # --- Variables for Logging ---
    pos_nominal = np.zeros(3)
    current_dist_min = 100.0
    current_vrel_at_min = 0.0
    
    
    
    print(f"Starting Simulation. Duration: 60s.")


    try:
        while t < 60.0:
            loop_start = time.perf_counter()

            if USE_BRIDGE:
                obs_pos, obs_vel, obs_acc = bridge.getObstacles()
            else:
                obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed=t)

            if len(obs_pos) == 0:
                # add a dummy obstacle far away to avoid empty lists
                obs_pos = [np.array([10.0, 10.0, 10.0])]
                obs_vel = [np.zeros(3)]
                obs_acc = [np.zeros(3)]
            



            # Task Space Trajectory
            goal_pose, nom_twist, nom_d_twist = planner_cart.getMotionLaw(trajectory_time % T_total)
            pos_nominal = goal_pose.translation.copy()

            # Forward Kinematics
            pin.framesForwardKinematics(model, data, q)
            pin.computeForwardKinematicsDerivatives(model, data, q, dq, np.zeros(model.nq))
            
            Tbt = data.oMf[tool_frame_id]
            x_curr = Tbt.translation
            tracking_error = np.linalg.norm(goal_pose.translation - x_curr)
            # Time Scaling ds(h, err)
            #Ds_target = np.clip(compute_ds_scaling_h(h_prev, tracking_error), 0, 1)
            
            # Time Scaling ds(h, err)
            
            Ds_target = np.clip(safety_utilis.compute_ds_scaling(current_dist_min, tracking_error, 0.1), 0, 1)
            
            DDtrajectory_time = 5.0 * (Ds_target - Dtrajectory_time)
                
            twist_goal = nom_twist * Dtrajectory_time
            goal_dtwist = (nom_d_twist * Dtrajectory_time**2 + nom_twist * DDtrajectory_time)
                
            # PD Control
            error_rot = Tbt.rotation @ pin.log3(Tbt.rotation.T @ goal_pose.rotation)
            twist_curr = pin.getFrameVelocity(model, data, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                
            acc_lin = Kp_tra * (goal_pose.translation - x_curr) + Kd_tra * (twist_goal[:3] - twist_curr.linear) + goal_dtwist[:3]
            acc_ang = Kp_rot * error_rot + Kd_rot * (twist_goal[3:] - twist_curr.angular) + goal_dtwist[3:]
            dtwist_des = np.hstack([acc_lin, acc_ang])
                
            # Jacobian Computation
            J = pin.computeFrameJacobian(model, data, q, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            dJ = pin.frameJacobianTimeVariation(model, data, q, dq, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jlin = J[:3, :]
            dJlin = dJ[:3, :]

            # QP Construction
            constraint_matrix = np.empty((0, model.nq + 1))
            constraint_vector = np.empty((0, 1))
            
            #Acceleration Constraints:
            constraint_acc_mat = np.hstack([np.eye(model.nq), np.zeros((model.nq, 1))])
            constraint_acc_vec = np.ones((model.nq, 1)) * DDq_MAX

            #Velocity constraints: -
            constraint_vel_mat = np.hstack([np.eye(model.nq), np.zeros((model.nq, 1))])
            upper_constraint_vel_vec = ((Dq_MAX - dq)/Tc).reshape(-1, 1)
            lower_constraint_vel_vec = ((-Dq_MAX - dq)/Tc).reshape(-1, 1)
            
            #Trajectory Tracking Constraints:
            next_trajectory_time =trajectory_time + Dtrajectory_time * Tc + 0.5 * DDtrajectory_time * Tc ** 2.0
            next_pose_des, _, _ = planner_cart.getMotionLaw(next_trajectory_time % T_total)
            next_x_des = next_pose_des.translation
            
            constraint_track_matrix_up = np.hstack([-Jlin * 0.5 * Tc**2, np.ones((3, 1))])
            constraint_track_matrix_lower = np.hstack([Jlin * 0.5 * Tc**2, np.ones((3, 1))])
     
            v_curr = Jlin @ dq
            x_free = x_curr + v_curr * Tc + 0.5*(dJlin@dq)*Tc**2
            
            lower_constraint_track_vector = (-eps_track - (x_free - next_x_des)).reshape(-1, 1)
            upper_constraint_track_vector = (-eps_track + (x_free - next_x_des)).reshape(-1, 1)
            
            
            # Delta Constraint:
            constraint_delta_matrix = np.zeros((1, model.nq + 1))
            constraint_delta_matrix[0, -1] = 1.0
            constraint_delta_vec = np.zeros((1, 1))

            
                
            h_min_curr = 100.0
            distance_vector = []
            vrel_vector = []
                
            for i in range(len(obs_pos)):
                    p_o = obs_pos[i]
                    v_o = obs_vel[i]
                    r = x_curr - p_o
                    dist = max(np.linalg.norm(r), 1e-6) #, print(f"Err: {error:.4f}, Scaling: {term_error:.4f}")
                    u_hr = r / dist
                    v_rel = np.dot(twist_curr.linear - v_o, u_hr)
                    vr_act = np.dot(twist_curr.linear, u_hr)
                    
            
                    distance_vector.append(dist)
                    vrel_vector.append(v_rel)
                    ## PFL CBF Evaluation [DEFINIZIONE A TRATTI]
                    #h_val = compute_h_PFL(dist, v_rel, v_rel_max, v_pfl, Tr_param, as_param)
                    #dh_dx = jacobian_h(dist, v_rel, v_rel_max, v_pfl, Tr_param, as_param)
                    
                    ## PFL CBF Evaluation [SOFTMAX METHOD]
                    h_softmax_val, dh_softmax_dx = safety_utilis.compute_h_softmax_and_grad(dist, v_rel)

                    ## Vmax CBF Evalutation
                    h_vmax_val, dh_vmax_dx = safety_utilis.compute_h_vmax_and_grad(vr_act)
                    
                    if h_softmax_val < h_min_curr: h_min_curr = h_softmax_val
                    
                    f_st, g_st =safety_utilis.range_state_derivative(twist_curr.linear, v_o)
                    
                    Jpsi_chi = safety_utilis.jacobian_psi(x_curr, p_o, twist_curr.linear, v_o)
                    
                    #Realizzazione vincoli QP per PFL
                    Lfh_softmax = dh_softmax_dx @ Jpsi_chi @ f_st
                    Lgh_softmax = dh_softmax_dx @ Jpsi_chi @ g_st
                    Apfl = np.hstack([(Lgh_softmax @ Jlin).reshape(1, -1), np.zeros((1, 1))])
                    Bpfl = (-Lgh_softmax @ dJlin @ dq - Lfh_softmax - gamma_param * h_softmax_val).reshape(1, -1)
                    
                    
                    #Realizzazione vincoli QP per Vmax
                    Lfh_vmax = dh_vmax_dx @ Jpsi_chi @ f_st
                    Lgh_vmax = dh_vmax_dx @ Jpsi_chi @ g_st
                    Avmax = np.hstack([(Lgh_vmax @ Jlin).reshape(1, -1), np.zeros((1, 1))])
                    Bvmax = (-Lgh_vmax @ dJlin @ dq - Lfh_vmax - gamma_param * h_vmax_val).reshape(1, -1)
                    
                    constraint_matrix = np.concatenate((constraint_matrix,
                                                        Apfl,
                                                        Avmax),
                                                        axis=0)
                    constraint_vector = np.concatenate((constraint_vector,
                                                        Bpfl,
                                                        Bvmax ),
                                                        axis=0)
                    

            # Add Joint Acceleration Limits And Tracking constraints
            constraint_matrix = np.concatenate((constraint_matrix,
                                                constraint_acc_mat, 
                                                -constraint_acc_mat,
                                                -constraint_vel_mat,
                                                constraint_vel_mat,
                                                constraint_track_matrix_up,
                                                constraint_track_matrix_lower,
                                                constraint_delta_matrix,
                                                ), axis=0)
            
            constraint_vector = np.concatenate((constraint_vector,
                                                -constraint_acc_vec,
                                                -constraint_acc_vec,
                                                -upper_constraint_vel_vec,
                                                lower_constraint_vel_vec,
                                                upper_constraint_track_vector,
                                                lower_constraint_track_vector, 
                                                constraint_delta_vec,
                                                ), axis=0)
                
            
                
            h_prev = h_min_curr
            dist_min_index = np.argmin(distance_vector)
            current_dist_min = distance_vector[dist_min_index]
            current_vrel_at_min = vrel_vector[dist_min_index]
            # Solve QP
            # Matrici per problema QP sull'accelerazione
            P_acc = J.T @ J + 1e-6 * np.eye(model.nq)
            b_acc = (J.T @ (dtwist_des - dJ @ dq)).flatten()
            ddq_nom = safety_utilis.damped_pinv_svd(J) @ (dtwist_des - dJ @ dq)
            
            # Matrici per problema QP su delta
            w_delta = 100.0; w_dyn_delta = 5000.0
            P_delta = np.array([[w_delta  +  w_dyn_delta]])
            b_delta = np.array([w_dyn_delta*delta])
            
            # Assemblaggio matrice P
            zeros_tr = np.zeros((model.nq, 1))
            zeros_bl = np.zeros((1, model.nq))
            
            P = np.block([
                [P_acc,    zeros_tr],
                [zeros_bl, P_delta]
            ])
            
            b = np.concatenate([b_acc, b_delta])
            
# =============================================================================
#             P = J.T @ J + 1e-6 * np.eye(model.nq)
#             b = (J.T @ (dtwist_des - dJ @ dq)).flatten()
#             ddq_nom = damped_pinv_svd(J) @ (dtwist_des - dJ @ dq)
# =============================================================================
            
            
            try:
                if len(obs_pos) > 0:
                    sol = quadprog.solve_qp(P, b, constraint_matrix.T, constraint_vector.flatten(), 0)
                    qp_sol = sol[0]
                    ddq = qp_sol[:model.nq]
                    delta = qp_sol[model.nq]
                    
                    #ddq, *_ = quadprog.solve_qp(P, b, constraint_matrix.T, constraint_vector.flatten(), 0)
                else:
                    ddq = ddq_nom
            except ValueError:
                print("QP infeasible")
                ddq = -10.0 * dq
                
            # Integration
            q += dq * Tc + 0.5 * ddq * Tc**2
            dq += ddq * Tc

            dq.clip(-1.0, 1.0, out=dq)
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
            print("\nSimulation ended. Generating Thesis Plots...")
            
            # 1. Raccogliamo tutti i log in un dizionario
            logs = {
                'time': log_time,
                'pos_act': log_pos_act,
                'pos_nom': log_pos_nom,
                'dist': log_dist,
                'vrel': log_vrel,
                'h': log_h,
                'ddq': log_ddq,
                'ddq_nom': log_ddq_nom,
                'ds_time': log_ds_time,
                'delta': log_delta
            }
            
            # 2. Raccogliamo i parametri usati nel controllo
            config = {
                'v_pfl': v_pfl,
                'a_s': a_s,
                'Tr': Tc,
                'v_max': v_rel_max
            }
            
            # 3. Inizializziamo il plotter e mostriamo tutto!
            plotter = ThesisPlotter(logs, config)
            plotter.show_all_plots()

if __name__ == "__main__":
    main()
