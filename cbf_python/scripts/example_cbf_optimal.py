# =============================================================================
# UR10 Kinematic Simulation with Pinocchio and Meshcat (threaded visual updates)
# =============================================================================
#
# This version spawns a background **daemon** thread that handles every visual
# operation (robot pose, moving obstacles, goal box, and HUD text).  The main
# 1 kHz control loop therefore never touches Meshcat directly, so its real‑time
# budget is preserved even on modest hardware.
#
# -----------------------------------------------------------------------------
#                      ***  CHANGES IN THIS REVISION  ***
# -----------------------------------------------------------------------------
# • `flush_visuals()` acquires `render_lock` **non‑blocking**; if the previous
#   visual push is still running we skip this frame instead of waiting.  This
#   prevents the control thread from stalling.
# • Completed the main loop, including the CBF/QP branch, joint‑space
#   integration, shared‑state publication, and fixed‑period sleep.
# • Added graceful keyboard‑interrupt handling: Ctrl‑C shuts down cleanly.
# -----------------------------------------------------------------------------
import os
import csv
import time

import meshcat.geometry as mgeom

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from scripts.util.joint_interpolator import SegmentedJointTrap
from scripts.util.visualization_daemon import VisualizationDaemon
from sharework import loadSharework

from scripts.util.test_utils import generate_obs_state, generate_velocity, compute_ee_pose, create_base_cfg, \
    bring_robot_home, plan_path, compute_cartesian_poses
from scripts.util.mean_visualizer import StochasticCBFVisualizer
import functools

from Controller.optimal_cbf_task_controller import BCFOptimalController, ControllerConfig

import math
from datetime import datetime
import rclpy

import signal
import threading
from scripts.util import csv_publishers, test_publish_utils as pub_utils
# from scripts.util.reference_xyz_trajectory import generate_cartesian_trajectory
from scripts.util.gaussian_process_util import read_config_data_from_csv
from scripts.util.bcf_utils import compute_dynamic_risk_index
from scripts.util.statistics_calculator import StatisticsCalculator

params_filename = "../params_csv/parameters_set.csv"
set_ID = "0"
duration = 15

SHOW_DATA = True
USE_BRIDGE = False
LOG_DATA = False
SAVE_DATA = False

parameters_type = "1"

stop_event = threading.Event()



h_cfg = "article"
v_cfg = "article"
# h_cfg = 1
# v_cfg = 1



test_name= "recorded_skeleton_23_h_-0.1_par"
# d_objective = 0.1

def _on_sigint_with_bridge(bridge, signum, frame):
    stop_event.set()
    try:
        bridge.shutdown()
    except Exception:
        pass

#signal.signal(signal.SIGINT, _on_sigint_with_bridge)

def main():
    # --------------------------- MODEL & VISUALS ---------------------------------
    log_path = "../resullts/simulation/scaling"
    # rclpy.init()



    home = np.array([90.0, -140.0, 140.0, -90.0, 90.0, 0.0]) * np.pi / 180.0

    UR10E_JOINTS = [
        "ur10e_shoulder_pan_joint",
        "ur10e_shoulder_lift_joint",
        "ur10e_elbow_joint",
        "ur10e_wrist_1_joint",
        "ur10e_wrist_2_joint",
        "ur10e_wrist_3_joint",
    ]
    model_wrapper = loadSharework(UR10E_JOINTS)
    prefix = 'ur10e_'
   

    # ------------------------ CONTROLLER SETUP -----------------------------------
    Tc =2e-3
    # cfg = create_base_cfg(set_ID, Tc, params_filename)
    cfg = ControllerConfig(Tc=Tc)
    delta = 4.5
    cfg.gamma =5.949803744662194
    cfg.lambda_acc = 1.4551402158959938e-10
    cfg.lambda_pos =  2098.0150948315577
    cfg.lambda_scaling = 16.558982747305556
    cfg.lambda_vel =   0.34298548889519453
    # read_config_data_from_csv(cfg, h_mean=h_cfg, v_mean=v_cfg, filename="../params_csv/log_best_trials.csv")
    cfg.delta_q_max[0:2] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta)
    cfg.delta_q_max[2:4] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 2
    cfg.delta_q_max[4:6] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 4

    # cfg.Dq_max = cfg.Dq_max*0.25
    # cfg.DDq_max = cfg.DDq_max*0.2
    print(cfg)

    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True, keypoint_to_log = -1)

    target_name = "ur10e_wrist_3_joint"
    idx = UR10E_JOINTS.index(target_name)
    if USE_BRIDGE:
        from Command_bridge.joint_command_bridge_modified import JointStateCommandBridge
        bridge = JointStateCommandBridge(
            ordered_joint_names=UR10E_JOINTS,
            threshold=1.1)  # radians (or native units)
        first_joint_position = bridge.wait_for_first_state( target_name, timeout=5.0)
        signal.signal(signal.SIGINT,
                  functools.partial(_on_sigint_with_bridge, bridge))
        if math.isnan(first_joint_position):
            bridge.shutdown()
            return
        first_joint_position = bridge.getPositions()
        bridge.switch_to_forward_position_controller_service()
    else:
        from Command_bridge.fake_command_bridge import FakeCommandBridge
        # Build camera pose from your INITI snippet
        quat = pin.Quaternion(0.83, 0.185, 0.513, 0.12)
        quat.normalize()

        R = quat.toRotationMatrix()
        if parameters_type == "0":
            T_wc = pin.SE3(R, np.array([-0.094, -0.93, 2.309]))
        else:
            T_wc = pin.SE3(R, np.array([1.04, -0.93, 2.309]))

        csv_path= ("../skeleton_vectors/skeleton_vectors_23.csv")        #csv_publishers.swap_csv(csv_in_path, csv_out_path, 7, 17)
        bridge = FakeCommandBridge(
            UR10E_JOINTS,
            csv_path=csv_path,
            Tworld_to_cam=T_wc,
            # slowdown_factor=0.1,
            slowdown_factor=1.0,
            t0= 0.0

        )
        rclpy.init()
        first_joint_position = home
    # ------------------------ PUBLISHER TARGETS  SETUP-----------------------------------
    if LOG_DATA:
        if USE_BRIDGE:
            joint_target_publisher = pub_utils.JointTargetPublisher(
                topic='joint_target',
                joint_names=UR10E_JOINTS,
                frame_id='world'
            )

            test_start_publisher = pub_utils.TestStartPublisher(
                topic='test_start'
            )
            cbf_out_publisher = pub_utils.DoubleArrayPublisher(
                topic='cbf_output',
                node_name='cbf_output_publisher',)
                # dim = 10)
            human_pos_publisher = pub_utils.DoubleArrayPublisher(
                topic='human_pos_keypoints',
                node_name='human_pos_publisher',)

            unfeasible_publisher =  pub_utils.DoubleArrayPublisher(
                topic='controller_status',
                node_name='ctrl_status_publisher',)
        else:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_path = log_path+"/"+str(now)
            # now = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
            print(test_path)
            os.makedirs(test_path, exist_ok = True)
            joint_target_publisher = csv_publishers.JointTargetCsvPublisher(
                csv_path= test_path + "/reference_trajectory.csv",
                column_names="time,target_joint_0_pos,target_joint_0_vel,target_joint_0_acceleration,target_joint_1_pos,target_joint_1_vel,target_joint_1_acceleration,target_joint_2_pos,target_joint_2_vel,target_joint_2_acceleration,target_joint_3_pos,target_joint_3_vel,target_joint_3_acceleration,target_joint_4_pos,target_joint_4_vel,target_joint_4_acceleration,target_joint_5_pos,target_joint_5_vel,target_joint_5_acceleration",
                joint_names=UR10E_JOINTS,
            )
            # JOINT STATE PUBLISHER ONLY FOR CSV LOGGING
            joint_state_publisher = csv_publishers.JointTargetCsvPublisher(
                csv_path =  test_path+"/joint_states.csv",
                column_names = "time,joint_0_pos,joint_0_vel,joint_0_acceleration,joint_1_pos,joint_1_vel,joint_1_acceleration,joint_2_pos,joint_2_vel,joint_2_acceleration,joint_3_pos,joint_3_vel,joint_3_acceleration,joint_4_pos,joint_4_vel,joint_4_acceleration,joint_5_pos,joint_5_vel,joint_5_acceleration",
                joint_names = UR10E_JOINTS,
            )

            test_start_publisher = csv_publishers.TestStartCsvPublisher(
                csv_path=test_path + "/TEST_START.csv",
                column_names="time,val"
            )
            cbf_out_publisher = csv_publishers.DoubleArrayCsvPublisher(
                csv_path=test_path + "/cbf_results.csv",
                column_names="time,h_min,d_min,trajectory_error,pos_ee_x,pos_ee_y,pos_ee_z,vel_ee_x,vel_ee_y,vel_ee_z,v_r_min,v_h_min,scaling")
            # dim = 10)
            human_pos_publisher = csv_publishers.DoubleArrayCsvPublisher(
                csv_path=test_path + "/human_positions.csv",
                column_names="time,human_keypoint_0_x,human_keypoint_0_y,human_keypoint_0_z,human_keypoint_1_x,human_keypoint_1_y,human_keypoint_1_z,human_keypoint_2_x,human_keypoint_2_y,human_keypoint_2_z,human_keypoint_3_x,human_keypoint_3_y,human_keypoint_3_z,human_keypoint_4_x,human_keypoint_4_y,human_keypoint_4_z,human_keypoint_5_x,human_keypoint_5_y,human_keypoint_5_z,human_keypoint_6_x,human_keypoint_6_y,human_keypoint_6_z,human_keypoint_7_x,human_keypoint_7_y,human_keypoint_7_z,human_keypoint_8_x,human_keypoint_8_y,human_keypoint_8_z,human_keypoint_9_x,human_keypoint_9_y,human_keypoint_9_z,human_keypoint_10_x,human_keypoint_10_y,human_keypoint_10_z,human_keypoint_11_x,human_keypoint_11_y,human_keypoint_11_z,human_keypoint_12_x,human_keypoint_12_y,human_keypoint_12_z,human_keypoint_13_x,human_keypoint_13_y,human_keypoint_13_z,human_keypoint_14_x,human_keypoint_14_y,human_keypoint_14_z,human_keypoint_15_x,human_keypoint_15_y,human_keypoint_15_z,human_keypoint_16_x,human_keypoint_16_y,human_keypoint_16_z,human_keypoint_17_x,human_keypoint_17_y,human_keypoint_17_z"
            )
            unfeasible_publisher = csv_publishers.DoubleArrayCsvPublisher(
                csv_path=test_path+'/controller_status.csv',
                column_names='time,status', )


    model = model_wrapper.model
    if SHOW_DATA:
        viz = MeshcatVisualizer(model, model_wrapper.collision_model, model_wrapper.visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()

    tmp = np.array([-300, 0., 0.])
    obstacle_positions = [tmp.copy() for _ in range(18*5)]
    tmp = np.array([0, 0., 0.])
    obstacle_velocities = [tmp.copy() for _ in range(18*5)]
    obstacle_accelerations = obstacle_velocities.copy()
    if SHOW_DATA:
        for i, pos in enumerate(obstacle_positions):
            if i == 7:
                viz.viewer[f"obstacle_{i}"].set_object(
                    mgeom.Sphere(0.1), mgeom.MeshLambertMaterial(color=0x000000)
                )
            else:
                viz.viewer[f"obstacle_{i}"].set_object(
                    mgeom.Sphere(0.1), mgeom.MeshLambertMaterial(color=0xFF0000)
                )

        # Goal box (green)
        side = 0.2
        viz.viewer["goal"].set_object(
            mgeom.Box([side, side, side / 10]), mgeom.MeshLambertMaterial(color=0x00FF00)
        )

        # HUD text node
        renderer = VisualizationDaemon(viz)  # default 60 Hz

    # --------------------------- CONTROL INITIALISATION --------------------------
    q = first_joint_position.copy()

    # ---------------------------TEST WAYPOINTS ------------------------------
    q = first_joint_position.copy()


    planner = SegmentedJointTrap(Dq_max=cfg.Dq_max*0.25, DDq_max=cfg.DDq_max*0.125)
    print("Computing trajectory...")
    # BRING THE ROBOT AT HOME BEFORE STARTING THE TEST
    if USE_BRIDGE:
        bring_robot_home(cfg, q, home, bridge, ctrl)
        q = home.copy()
    # 2 · add way‑points -------------------------------------------
    plan_path(planner,q)
    n_wp = 10
    cartesian_configs = compute_cartesian_poses(q, model)

    T_total = planner.computeTime()
    print(f"Total time: {T_total}")
    min_dist = []
    if SHOW_DATA:
        renderer.publishPath(planner.publishPath())

    # Instantiate the StatisticsCalculator
    stats_calculator = StatisticsCalculator(
        n_wp=n_wp,
        T_total=T_total,
        cartesian_configs=cartesian_configs,
        Tc=Tc,
        scaling_threshold=0.5 # Use the same default as in the class
    )

    # Remove old statistics accumulation variables
    # ct, ct_qp, ct_ssm, ct_planner, ct_pin, h_log, trj_error_log, scaling_log, traj_cart_error_log, s_index_log = [], [], [], [], [], [], [], [], [], []
    # lap_count = 0
    # on_target_count = 0
    # prec_target = -1
    # enable_lap_count = True
    # unfeasible_cnt = 0
    # timeout_cycles = cycles = 0
    # violations = sum_scale = trajectory_error_sum = trajectory_cart_error_sum= 0
    # low_scale_count = 0
    # scaling_threshold = 0.5 # This is now passed to the StatisticsCalculator

    # ------------------------------ MAIN LOOP -------------------- ----------------
    if LOG_DATA:
        test_start_publisher.publish_once(True) # pyright: ignore[reportPossiblyUnboundVariable]
    try:

        t = 0.0
        trajectory_time = 0.0

        ctrl.reset_state(q)
        visualizer = StochasticCBFVisualizer()
        while t < duration and not stop_event.is_set():
            h_min = np.inf

            loop_start = time.perf_counter()

            if USE_BRIDGE:
                obstacle_positions, obstacle_velocities, obstacle_accelerations = bridge.getObstacles()
            else:
                obstacle_positions, obstacle_velocities, obstacle_accelerations = bridge.getObstacles(elapsed=t)
            # print ("obstacle_positions:", obstacle_positions[7])
            # print ("type(obstacle_positions):", type(obstacle_positions))
            # print("size(obstacle_positions): ", obstacle_positions.shape)
            # cycles += 1

            nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time % T_total)
            # if obstacle_positions.shape[0] > 7:
            #     out = ctrl.step(
            #         obs_pos=obstacle_positions[7].reshape(1,3),
            #         obs_vel=obstacle_velocities[7].reshape(1,3),
            #         obs_acc=obstacle_accelerations[7].reshape(1,3),
            #         nominal_q=nominal_q,
            #         nominal_Dq=nominal_Dq,
            #         nominal_DDq=nominal_DDq
            #     )
            # else:
            out = ctrl.step(
                obs_pos=obstacle_positions,
                obs_vel=obstacle_velocities,
                obs_acc=obstacle_accelerations,
                nominal_q=nominal_q,
                nominal_Dq=nominal_Dq,
                nominal_DDq=nominal_DDq
            )
            unfeasible_string = out["unfeasible_cnt"]
            q = out["q"]

            if stats_calculator.cycles < 5: # Use stats_calculator.cycles for initial prints
                print(f"q pln={nominal_q.T}\nq act={q.T}")
            dq = out["dq"]
            ddq = out["ddq"]
            trajectory_time = out["trajectory_time"]
            Dtrajectory_time = out["Dtrajectory_time"]
            
            # Old lap count logic removed, now handled by StatisticsCalculator
            # if (trajectory_time % T_total) < Tc:
            #     if enable_lap_count:
            #         lap_count += 1
            #         prec_target = -1
            #         enable_lap_count = False
            # else:
            #     enable_lap_count = True
            
            # ct_qp.append(elapsed) # This was for a specific timing, might need to be re-evaluated if still needed

            # --------------------------- INTEGRATION ----------------------------
            t += Tc
            end_eff_pos = out["end_effector_pos"]
            end_eff_nominal_pos = out["Tbt_nominal"].translation
            trajectory_cart_err = float(np.linalg.norm(end_eff_pos - end_eff_nominal_pos))
            end_eff_vel = out["end_effector_vel"]

            if stats_calculator.cycles % 5000 == 0: # Use stats_calculator.cycles
                print(f"STILL ALIVE! T: {t:.2f}s")
            if USE_BRIDGE and not stop_event.is_set():
                bridge.sendCommand(q)
            if not stop_event.is_set() and LOG_DATA:
                joint_target_publisher.publish_once(t, nominal_q, nominal_Dq, nominal_DDq) # pyright: ignore[reportPossiblyUnboundVariable]
                hmin = out["h_min"]
                dmin = out["d_min"]
                trj_error = out["trajectory_error"]
                vr_min = out["vr_min"]
                vh_min = out["vh_min"]
                scaling = out["Dtrajectory_time"]
                cbf_out_publisher.publish_once(
                    t,
                    [
                        hmin,
                        dmin,
                        trj_error,
                        end_eff_pos[0],
                        end_eff_pos[1],
                        end_eff_pos[2],
                        end_eff_vel[0],
                        end_eff_vel[1],
                        end_eff_vel[2],
                        vr_min,
                        vh_min,
                        scaling,
                    ]
                ) # pyright: ignore[reportPossiblyUnboundVariable]
                human_pos_publisher.publish_once(t, obstacle_positions)
                ctrl_status_code = -99999
                if unfeasible_string == "FEASIBLE":
                    ctrl_status_code = 0
                elif unfeasible_string == "RECOVERING":
                    ctrl_status_code = 1
                elif unfeasible_string == "UNFEASIBLE":
                    ctrl_status_code = 2
                unfeasible_publisher.publish_once(t, [ctrl_status_code])
            if not USE_BRIDGE and LOG_DATA:
                joint_state_publisher.publish_once(t, q, dq, ddq)
            # ----------------------------- TIMING -------------------------------
            # Old min_dist, violations, sum_scale, trajectory_error_sum, trajectory_cart_error_sum, low_scale_count logic removed
            # dist = []
            # for i in range(len(cartesian_configs.values())):
            #     q_wp = list(cartesian_configs.values())[i]
            #     dist.append(np.linalg.norm(q_wp - end_eff_pos))
            #     if  np.linalg.norm(q_wp - end_eff_pos) < 2e-03 and prec_target != i:
            #         on_target_count += 1
            #         prec_target = i
            #         break
            # if np.min(dist) > 0.0:
            #     min_dist.append(np.min(dist))
            # if out["h_min"] < 0 and out["vr_min"] < -1e-3:
            #     violations += 1
            # sum_scale += out["Dtrajectory_time"]
            # trajectory_error_sum += out["trajectory_error"]
            # trajectory_cart_error_sum += trajectory_cart_err
            # if (Dtrajectory_time) < scaling_threshold:
            #     low_scale_count += 1
            visualizer.update_vectors(out["h_min"], out["d_min"], out["vr_min"]-out["vh_min"], t,)
            
            elapsed = time.perf_counter() - loop_start

            s_index = None
            if stats_calculator.cycles > 1: # s_index requires at least 2 cycles of data
                s_index = compute_dynamic_risk_index(end_eff_pos=end_eff_pos, end_eff_vel=end_eff_vel,
                                                              obs_positions=obstacle_positions, obs_velocities=obstacle_velocities,
                                                              obs_accelerations=obstacle_accelerations, a_s=cfg.a_s)

            # Update the statistics calculator
            stats_calculator.update(
                out=out,
                trajectory_cart_err=trajectory_cart_err,
                s_index=s_index,
                elapsed_time=elapsed,
                unfeasible_string=unfeasible_string,
                end_eff_pos=end_eff_pos
            )

            # Old ct.append, s_index_log, scaling_log, h_log, trj_error_log, traj_cart_error_log removed
            # if cycles>1:
            #     ct.append(elapsed)
            #     s_index = compute_dynamic_risk_index_with_acc(end_eff_pos=end_eff_pos, end_eff_vel=end_eff_vel,
            #                                                   obs_positions=obstacle_positions, obs_velocities=obstacle_velocities,
            #                                                   obs_accelerations=obstacle_accelerations, a_s=cfg.a_s)
            #     s_index_log.append(s_index)
            #     scaling_log.append(Dtrajectory_time)
            #     h_log.append(out["h_min"])
            #     trj_error_log.append(out["trajectory_error"])
            #     traj_cart_error_log.append(trajectory_cart_err)
            
            rest = Tc - elapsed
            # print(ctrl.cfg.delta_q_max)
            if rest > 0:
                if SHOW_DATA:
                    vizualization_string =f"h={out['h_min']:.2f}m  scale={out['Dtrajectory_time']:.3f}  err={out['trajectory_error']:.2f} ctrl_state:{unfeasible_string}"

                    renderer.push_state(out["q"], out["Tbt_nominal"], out["obs_pos"], viz_string = vizualization_string)
                    elapsed = time.perf_counter() - loop_start
                    rest = max(0.0,Tc - elapsed)
                    time.sleep(rest)
                else:
                    time.sleep(0.0001)
            else:
                # Old timeout_cycles increment removed, now handled by StatisticsCalculator
                # timeout_cycles+=1
                pass
            # Old unfeasible_cnt increment removed, now handled by StatisticsCalculator
            # if unfeasible_string != "FEASIBLE":
            #     unfeasible_cnt += 1

        if not stop_event.is_set() and LOG_DATA:
            test_start_publisher.publish_once(False) # pyright: ignore[reportPossiblyUnboundVariable]

    except KeyboardInterrupt:
        # request a graceful stop; loop condition will exit on next iteration
        stop_event.set()
# 
    finally:
        try:
            pub_utils.publish_test_start_once(False)
        except Exception as e:
            print(f"[shutdown] one-shot publish failed: {e}")
    
    # Print statistics using the new class
    print(stats_calculator)

    # Old statistics calculation and printing removed
    # computation_times = np.array(ct)
    # scaling_log = np.array(scaling_log)
    # h_log = np.array(h_log)
    # trj_error_log = np.array(trj_error_log)
    # traj_cart_error_log = np.array(traj_cart_error_log)
    # print(f"LAP COUNT: {lap_count}")

    # on_target_rate = on_target_count/(n_wp * ((lap_count)+ ((trajectory_time % T_total)/T_total)))
    # lap_count = lap_count + ((trajectory_time % T_total)/T_total)
    # print(f"average scaling = {np.mean(scaling_log)}")

    # stats = {
    #     "computation_times": computation_times,
    # }

    # trj_error_diff = np.abs(np.diff(trj_error_log))
    # total_variation_error = np.sum(trj_error_diff)

    # trajectory_cart_error_diff = np.abs(np.diff(traj_cart_error_log))
    # total_variation_cart_error = np.sum(trajectory_cart_error_diff)
    # mean_tv_error = total_variation_error / max(1, cycles)
    # mean_tv_cartesian = total_variation_cart_error / max(1, cycles)

    # on_target_rate = on_target_count / (n_wp * ((lap_count) + ((trajectory_time % T_total) / T_total)))
    # lap_count = lap_count + ((trajectory_time % T_total) / T_total)
    # viol_rate = violations / max(1, cycles)
    # mean_scale = sum_scale / max(1, cycles)
    # mean_trajectory_error = trajectory_error_sum / max(1, cycles)
    # mean_cartesian_error = trajectory_cart_error_sum / max(1, cycles)
    # low_scale_rate = low_scale_count / max(1, cycles)

    # print(f"timeout cycles = {timeout_cycles} over {cycles}, percentage = {100.0*timeout_cycles/cycles}, average = {np.mean(computation_times)}")
    # print(f"unfeasible cycles = {unfeasible_cnt} over {cycles}, percentage = {100.0*unfeasible_cnt/cycles}")
    # print(f"LAP COUNT: {lap_count}")
    # print("on target count: ", on_target_count)
    # print(f"WAYPOINTS REACHING PERCENTAGE: {on_target_rate*100.0} %")
    # print(f"VIOLATION RATE: {viol_rate*100} %")
    # print(f"MEAN SCALING: {mean_scale}")
    # print(f"MEAN TRAJECTORY ERROR: {mean_trajectory_error}")
    # print(f"LOW SCALE RATE: {low_scale_rate*100}")
    # print(f"MEAN CARTESIAN ERROR: {mean_cartesian_error}")
    # print(f"MEAN TV JOINT ERROR: {mean_tv_error*1000}")
    # print(f"MEAN TV CARTESIAN ERROR: {mean_tv_cartesian*1000}")
    # print(f"MEAN RISK INDEX : {sum(s_index_log)/len(s_index_log)}")
    
    visualizer.compute_mean_cov(True)
    # CREATING CARTESIAN REFERENCE CSV FILE
    if LOG_DATA:
        if USE_BRIDGE:
            folder_name = "" # UPDATE WITH THE PATH THE TRAJECTORY LOGGER NODE USES
        else:
            folder_name = test_path
        # generate_cartesian_trajectory(folder_name+"/") # This function is commented out in imports, so keeping it commented here

    # SAVING RESULTS
    if SAVE_DATA:
        file_path = '../resullts/simulation_data.csv'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Intestazioni delle colonne (headers)
        headers = [
            "test_type",
            "lambda_pos",
            "lambda_vel",
            "lambda_scaling",
            "lambda_acc",
            "delta",
            "gamma",
            'on_target_rate',
            'lap_count',
            'viol_rate',
            'mean_scale',
            'mean_trajectory_error',
            "low_scale_rate"
        ]

        # Get stats from the calculator
        final_stats = stats_calculator._calculate_stats()

        # I dati da salvare (calcolati come nel tuo esempio)
        row_data = {
            "test_type": test_name,
            "lambda_pos": cfg.lambda_pos,
            "lambda_vel": cfg.lambda_vel,
            "lambda_scaling": cfg.lambda_scaling,
            "lambda_acc": cfg.lambda_acc,
            "delta": delta,
            "gamma": cfg.gamma,
            'on_target_rate': final_stats['on_target_rate'],
            'lap_count': final_stats['lap_count'],
            'viol_rate': final_stats['violation_rate'],
            'mean_scale': final_stats['mean_scaling'],
            'mean_trajectory_error': final_stats['mean_trajectory_error'],
            "low_scale_rate": final_stats['low_scale_rate']
        }

        # Controllo se il file esiste già per scrivere l'header solo la prima volta
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)

            # Se il file è nuovo, scriviamo l'intestazione
            if not file_exists:
                writer.writeheader()

            # Aggiungiamo la riga con i risultati
            writer.writerow(row_data)
if __name__ == "__main__":
    main()
