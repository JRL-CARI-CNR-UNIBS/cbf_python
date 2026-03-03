#####MODIFICHE RISPETTO ALLO SCRIPT ORIGINALE: 
#Aumentiamo le dimensioni del vettore degli stati X = [d, vrel , vr_act] in modo da poter definire un vincolo di velocità massimo sensibile alla sola velocità del robot
#questo comporta delle modifiche al calcolo dello jacobiano di h e di psi



import time
import math
import numpy as np
import pinocchio as pin
import meshcat.geometry as mgeom
import quadprog
import matplotlib.pyplot as plt

from pinocchio.visualize import MeshcatVisualizer
from visualization_daemon import VisualizationDaemon
from sharework import loadSharework
# from optimal_cbf_task_controller import BCFOptimalController, ControllerConfig

from interpolator import SegmentedSE3Trap
from joint_interpolator import SegmentedJointTrap
from pinocchio import SE3

# CONFIGURATION
USE_BRIDGE = False

# Safety Parameters
C_param = 0.25
Tr_param = 0.15
as_param = 2.5
gamma_param = 30.0
Tc = 2e-3
DDq_MAX = np.pi**2*5


eps_track = 0.03 # 3cm
rho = 20.0 #parametro softmin e softmax


# MATH & KINEMATICS
# =============================================================================
def compute_h_softmax_and_grad(d, v_rel, Tr, a_s, v_pfl, rho):
    h_br = d - (-v_rel*Tr + (v_rel**2) / (2.0 * abs(a_s)))
    grad_br = np.array([1.0,Tr -v_rel / a_s, 0.0])
    
    h_pfl = (v_pfl + v_rel) * Tr
    grad_pfl = np.array([0.0, Tr, 0.0])

    max_inner = max(h_br, h_pfl)
    
    exp_br = np.exp(rho * (h_br - max_inner))
    exp_pfl = np.exp(rho * (h_pfl - max_inner))
    sum_inner = exp_br + exp_pfl
    
    h_softmax = max_inner + (1.0 / rho) * np.log(sum_inner)
    
    # Pesi Interni
    omega_br = exp_br / sum_inner
    omega_pfl = exp_pfl / sum_inner
    
    grad_hsoftmax = omega_br * grad_br + omega_pfl * grad_pfl
    
    return h_softmax, grad_hsoftmax

def compute_h_vmax_and_grad(v_rel_max, vr_act):
    h_vmax_pos = (v_rel_max - vr_act)*(v_rel_max + vr_act)
    grad_vmax = np.array([0.0, 0.0, -2.0 * vr_act ])
    
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
    
    row_d = np.hstack((u_rh.T, 
                       -u_rh.T,
                        np.zeros((1, 3)),
                        np.zeros((1, 3))))
    
    row_vrel = np.hstack((wP_over_d.reshape(1, -1),
                        -wP_over_d.reshape(1, -1), 
                        u_rh.T,
                        -u_rh.T))
    
    row_vract = np.hstack((vrP_over_d.reshape(1, -1),
                        -vrP_over_d.reshape(1, -1), 
                        u_rh.T,
                        np.zeros((1, 3))))
    
    return np.vstack((row_d, row_vrel, row_vract))

def damped_pinv_svd(J, lam=1e-4):
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_damped = S / (S ** 2 + lam ** 2)
    return (Vt.T * S_damped) @ U.T

def pose_eul(z, y, x, xyz):
    R = pin.utils.rotate('z', z) @ pin.utils.rotate('y', y) @ pin.utils.rotate('x', x)
    return SE3(R, np.array(xyz))

def compute_ds_scaling_h(h, error):
    # Fattore Sicurezza (Sigmoide)
    h_threshold = 0.9
    slope_h = 30.0
    term_safety = 1.0 / (1.0 + np.exp(-slope_h * (h - h_threshold)))
    
    # Fattore Errore
    sigma_error = eps_track #m di tolleranza
    term_error = np.exp(- (error**2) / (2 * sigma_error**2))
    
    ds = min(term_safety, term_error)
    return ds

def compute_ds_scaling_d(distance, error):
    d_activation = 0.2
    slope_d = 100.0
    term_safety = 1.0 / (1.0 + np.exp(-slope_d * (distance - d_activation)))
    #print(f"Distance: {distance:.4f}, Scaling-safety-term: {term_safety:.4f}")

    limit_err = eps_track * 1.5  
    n_power = 10.0               

    term_error = np.exp(- (abs(error) / limit_err)**n_power)

    # print(f"Err: {error:.4f}, Scaling: {term_error:.4f}")

    ds = min(term_safety, term_error)
    return ds


# MAIN
def main():
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

        T_wc = pin.SE3(R, np.array([0.208, -0.883, 2.351]))
        csv_path = "/home/nyquist/projects/cells_ws/src/zed_skeleton_kinematics/csv_files/skeleton_vectors_22.csv"
        #csv_path = "/home/nyquist/projects/cells_ws/src/zed_skeleton_kinematics/csv_files/skeleton_vectors_14_NORMAL_TEST1.csv"
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
    planner_cart = SegmentedSE3Trap(vlin_max=0.6, vang_max=1.2, alin_max=1.8, aang_max=2.0)
    
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
    
    v_rel_max = 2.5;
    v_pfl = 0.0;
    

    # --- Variables for Logging ---
    pos_nominal = np.zeros(3)
    current_dist_min = 100.0
    current_vrel_at_min = 0.0
    
    
    
    print(f"Starting Simulation. Duration: 150s.")


    try:
        while t < 150.0:
            loop_start = time.perf_counter()

            if USE_BRIDGE:
                obs_pos, obs_vel, obs_acc = bridge.getObstacles()
            else:
                obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed=t)

            
            



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
            Ds_target = np.clip(compute_ds_scaling_d(current_dist_min, tracking_error), 0, 1)
            
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
            
            constraint_acc_mat = np.hstack([np.eye(model.nq), np.zeros((model.nq, 1))])
            constraint_acc_vec = np.ones((model.nq, 1)) * DDq_MAX
            
            
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
                    h_softmax_val, dh_softmax_dx = compute_h_softmax_and_grad(dist, v_rel, Tr_param, as_param, v_pfl, rho)
                    
                    
                    ## Vmax CBF Evalutation
                    h_vmax_val, dh_vmax_dx = compute_h_vmax_and_grad(v_rel_max, vr_act)
                    
                    if h_softmax_val < h_min_curr: h_min_curr = h_softmax_val
                    
                    f_st, g_st = range_state_derivative(twist_curr.linear, v_o)
                    
                    Jpsi_chi = jacobian_psi(x_curr, p_o, twist_curr.linear, v_o)
                    
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
                                                constraint_track_matrix_up,
                                                constraint_track_matrix_lower,
                                                constraint_delta_matrix,
                                                ), axis=0)
            
            constraint_vector = np.concatenate((constraint_vector,
                                                -constraint_acc_vec,
                                                -constraint_acc_vec,
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
            ddq_nom = damped_pinv_svd(J) @ (dtwist_des - dJ @ dq)
            
            # Matrici per problema QP su delta
            w_delta = 100.0
            P_delta = np.array([[w_delta]])
            b_delta = np.array([0.0])
            
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
        # --- PLOTs ---
        if len(log_time) > 0:
            time_arr = np.array(log_time)
            ds_time_arr = np.array(log_ds_time)
            pos_act_arr = np.array(log_pos_act)
            pos_nom_arr = np.array(log_pos_nom)
            dist_arr = np.array(log_dist)
            vrel_arr = np.array(log_vrel)
            ddq_arr = np.array(log_ddq)
            ddq_nom_arr = np.array(log_ddq_nom)
            delta_arr = np.array(log_delta)
            # Fig: XYZ Position Tracking
            fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            labels = ['x [m]', 'y [m]', 'z [m]']
            for i in range(3):
                axs1[i].plot(time_arr, pos_act_arr[:, i], 'r-', label='Actual')
                axs1[i].plot(time_arr, pos_nom_arr[:, i], 'k--', label='Nominal')
                axs1[i].set_ylabel(labels[i])
                axs1[i].grid(True)
            axs1[0].legend(loc='upper right')
            axs1[0].set_title('Task Space Position Tracking')
            axs1[2].set_xlabel('Time [s]')
            plt.tight_layout()

            # Fig: Distance and V_rel
            fig2, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Min Distance [m]', color='r')
            ax1.plot(time_arr, dist_arr, 'r-', label='Distance')
            ax1.tick_params(axis='y', labelcolor='r')
            ax1.grid(True)
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('V_rel [m/s]', color='g')
            ax2.plot(time_arr, vrel_arr, 'g-', label='V_rel')
            ax2.axhline(y=-v_pfl, color='k', linestyle=':', label=f'-v_PFL ({-v_pfl})')
            ax2.tick_params(axis='y', labelcolor='g')
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            plt.title('Obstacle: Min Distance & Relative Velocity')
            plt.tight_layout()

            # Fig: CBF 
            plt.figure(figsize=(10, 6))
            plt.plot(time_arr, log_h, 'r', label='h_min')
            plt.axhline(0, color='k', linestyle='--')
            plt.ylabel('h value')
            plt.title('Control Barrier Function')
            plt.legend()
            plt.grid(True)
            

            # Fig: Joint Accelerations
            plt.figure(figsize=(10, 6))
            #colors = ['r', 'g', 'b', 'c', 'm', 'y']
            #for j in range(model.nq):
               # plt.plot(time_arr, ddq_arr[:, j], color=colors[j % len(colors)], label=f'ddq_{j}')
            plt.plot(time_arr, ddq_arr[:,-1], label=f'ddq')
            plt.plot(time_arr, ddq_nom_arr[:,-1], label = f'ddq_des' )
            plt.xlabel('Time [s]')
            plt.ylabel('Joint Acc [rad/s^2]')
            plt.title('Joint Accelerations')
            plt.legend(ncol=3)
            plt.grid(True)
            plt.tight_layout()
            
            
            # Fig: DS_Time Scaling
            plt.figure(figsize=(10, 6))
            plt.plot(time_arr, ds_time_arr , 'r', label='ds_traj_time')
            plt.axhline(0, color='k', linestyle='--')
            plt.xlabel('time')
            plt.ylabel('DS_traj_time')
            plt.title('DS_traj_time')
            plt.legend()
            plt.grid(True)
            
            # Fig: Delta
            plt.figure(figsize=(10, 6))
            plt.plot(time_arr, delta_arr , 'r' )
            plt.axhline(0, color='k', linestyle='--')
            plt.xlabel('time')
            plt.ylabel('Delta')
            plt.title('Delta')
            plt.legend()
            plt.grid(True)
            
            

            plt.show()

if __name__ == "__main__":
    main()
