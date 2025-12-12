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
import time

import meshcat.geometry as mgeom

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from joint_interpolator import SegmentedJointTrap
from visualization_daemon import VisualizationDaemon
from sharework import loadSharework

from bcf_utils import make_summary_figure, print_stats_table

import functools

from optimal_cbf_task_controller import BCFOptimalController, ControllerConfig

import math
from datetime import datetime
import test_publish_utils as pub_utils
import rclpy

import signal
import threading
import csv_publishers

stop_event = threading.Event()

def _on_sigint_with_bridge(bridge, signum, frame):
    stop_event.set()
    try:
        bridge.shutdown()
    except Exception:
        pass

#signal.signal(signal.SIGINT, _on_sigint_with_bridge)

def main():
    # --------------------------- MODEL & VISUALS ---------------------------------
    USE_BRIDGE = True
    LOG_DATA = True
    log_path = "resullts/simulation/scaling"
    # rclpy.init()


    duration = 150.0

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
    cfg = ControllerConfig(Tc=Tc)
    cfg.lambda1 = 500.0
    cfg.lambda2 = 2.55
    cfg.lambda3 = 130#1.0e3
    cfg.lambda4 = 4.0e-05 
    cfg.delta_q_max[0:2] = np.deg2rad(np.array([1,1], dtype=np.float64) * 1.5)*3
    cfg.delta_q_max[2:4] = np.deg2rad(np.array([1,1], dtype=np.float64) * 3)*3
    cfg.delta_q_max[4:6] = np.deg2rad(np.array([1,1], dtype=np.float64) * 6)*3
    cfg.gamma = 10.0
    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg)

    target_name = "ur10e_wrist_3_joint"
    idx = UR10E_JOINTS.index(target_name)
    if USE_BRIDGE:
        from joint_command_bridge_modified import JointStateCommandBridge
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
        from fake_command_bridge import FakeCommandBridge
        # Build camera pose from your INITI snippet
        quat = pin.Quaternion(0.83, 0.185, 0.513, 0.12)
        quat.normalize()

        R = quat.toRotationMatrix()

        T_wc = pin.SE3(R, np.array([0.094, -0.93, 2.309]))

        bridge = FakeCommandBridge(
            UR10E_JOINTS,
            csv_path="/home/galileo/Desktop/skeleton_vectors_19.csv",
            Tworld_to_cam=T_wc,
            # slowdown_factor=0.1,
            slowdown_factor=1.0,

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
        else:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_path = log_path+"/"+str(now)
            now = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
            print(test_path)
            os.makedirs(test_path, exist_ok = True)
            joint_target_publisher = csv_publishers.JointTargetCsvPublisher(
                csv_path=test_path + "/reference_trajectory" + now + ".csv",
                column_names="time,target_joint_0_pos,target_joint_0_vel,target_joint_0_acceleration,target_joint_1_pos,target_joint_1_vel,target_joint_1_acceleration,target_joint_2_pos,target_joint_2_vel,target_joint_2_acceleration,target_joint_3_pos,target_joint_3_vel,target_joint_3_acceleration,target_joint_4_pos,target_joint_4_vel,target_joint_4_acceleration,target_joint_5_pos,target_joint_5_vel,target_joint_5_acceleration",
                joint_names=UR10E_JOINTS,
            )
            # JOINT STATE PUBLISHER ONLY FOR CSV LOGGING
            joint_state_publisher = csv_publishers.JointTargetCsvPublisher(
                csv_path =  test_path+"/joint_states"+now+".csv",
                column_names = "time,joint_0_pos,joint_0_vel,joint_0_acceleration,joint_1_pos,joint_1_vel,joint_1_acceleration,joint_2_pos,joint_2_vel,joint_2_acceleration,joint_3_pos,joint_3_vel,joint_3_acceleration,joint_4_pos,joint_4_vel,joint_4_acceleration,joint_5_pos,joint_5_vel,joint_5_acceleration",
                joint_names = UR10E_JOINTS,
            )

            test_start_publisher = csv_publishers.TestStartCsvPublisher(
                csv_path=test_path + "/TEST_START"+now+".csv",
                column_names="time,val"
            )
            cbf_out_publisher = csv_publishers.DoubleArrayCsvPublisher(
                csv_path=test_path + "/cbf_results"+now+".csv",
                column_names="time,h_min,d_min,trajectory_error,pos_ee_x,pos_ee_y,pos_ee_z,vel_ee_x,vel_ee_y,vel_ee_z,v_r_min,v_h_min,scaling")
            # dim = 10)
            human_pos_publisher = csv_publishers.DoubleArrayCsvPublisher(
                csv_path=test_path + "/human_positions"+now+".csv",
                column_names="time,human_keypoint_0_x,human_keypoint_0_y,human_keypoint_0_z,human_keypoint_1_x,human_keypoint_1_y,human_keypoint_1_z,human_keypoint_2_x,human_keypoint_2_y,human_keypoint_2_z,human_keypoint_3_x,human_keypoint_3_y,human_keypoint_3_z,human_keypoint_4_x,human_keypoint_4_y,human_keypoint_4_z,human_keypoint_5_x,human_keypoint_5_y,human_keypoint_5_z,human_keypoint_6_x,human_keypoint_6_y,human_keypoint_6_z,human_keypoint_7_x,human_keypoint_7_y,human_keypoint_7_z,human_keypoint_8_x,human_keypoint_8_y,human_keypoint_8_z,human_keypoint_9_x,human_keypoint_9_y,human_keypoint_9_z,human_keypoint_10_x,human_keypoint_10_y,human_keypoint_10_z,human_keypoint_11_x,human_keypoint_11_y,human_keypoint_11_z,human_keypoint_12_x,human_keypoint_12_y,human_keypoint_12_z,human_keypoint_13_x,human_keypoint_13_y,human_keypoint_13_z,human_keypoint_14_x,human_keypoint_14_y,human_keypoint_14_z,human_keypoint_15_x,human_keypoint_15_y,human_keypoint_15_z,human_keypoint_16_x,human_keypoint_16_y,human_keypoint_16_z,human_keypoint_17_x,human_keypoint_17_y,human_keypoint_17_z"
            )


    model = model_wrapper.model
    viz = MeshcatVisualizer(model, model_wrapper.collision_model, model_wrapper.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    tmp = np.array([-300, 0., 0.])
    obstacle_positions = [tmp.copy() for _ in range(18*5)]
    tmp = np.array([0, 0., 0.])
    obstacle_velocities = [tmp.copy() for _ in range(18*5)]
    obstacle_accelerations = obstacle_velocities.copy()

    for i, pos in enumerate(obstacle_positions):
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
    # q2 = home.copy()
    # q2[1] = -np.pi * 0.5
    # q2[2] = np.pi * 0.5
    # q3 = np.array([ 40.0, -80.0, 100.0, -120.0, 90.0, 0.0])*np.pi/180.0
    # q4 = np.array([ 122.0, -70.0, 100.0, -120.0, 90.0, 0.0])*np.pi/180.0
    # print(f"q={q.T}\nq={q2.T}")

    # ---------------------------TEST WAYPOINTS ------------------------------
    q10 = np.array([ 31.0, -78.0, 115.0, -127.0, 86.0, -32.0])*np.pi/180.0
    q20 =  np.array([ 31.0, -83.0, 98.0, -110.0, 86.0, -32.0])*np.pi/180.0
    q22 =  np.array([ 40.0, -126.0, 141.0, -100.0, 86.0, 45.0])*np.pi/180.0
    q25 =  np.array([ 130.0, -100.0, 125.0, -115.0, 94.0, -20.0])*np.pi/180.0
    q30 =  np.array([ 136.0, -60.0, 90.0, -122.0, 90.0, 45.0])*np.pi/180.0
    q40 =  np.array([ 134.0, -65.0, 70.0, -90.0, 90.0, 45.0])*np.pi/180.0
    cfg.Dq_max = cfg.Dq_max*0.25
    cfg.DDq_max = cfg.DDq_max*0.2
    planner = SegmentedJointTrap(Dq_max=cfg.Dq_max*0.25, DDq_max=cfg.DDq_max*0.25)

    # 2 · add way‑points -------------------------------------------
    for _ in range(3):
        planner.addWayPoint(q)
        planner.addWayPoint(q10)
        planner.addWayPoint(q20)
        planner.addWayPoint(q10)
        planner.addWayPoint(q22)
        planner.addWayPoint(q25)
        planner.addWayPoint(q30)
        planner.addWayPoint(q40)
        planner.addWayPoint(q30)
        planner.addWayPoint(q)
    T_total = planner.computeTime()
    print(f"Total time: {T_total}")
    renderer.publishPath(planner.publishPath())

    ct, ct_qp, ct_ssm, ct_planner, ct_pin, h_log, trj_error_log, scaling_log = [], [], [], [], [], [], [], []

    # ------------------------------ MAIN LOOP -------------------- ----------------

    test_start_publisher.publish_once(True) # pyright: ignore[reportPossiblyUnboundVariable]
    unfeasible_cnt = 0
    try:

        t = 0.0

        trajectory_time = 0.0


        timeout_cycles = cycles =0

        ctrl.reset_state(q)
        # test_start = True
        while t < duration and not stop_event.is_set():
            # if t%T_total == 0:
            #     test_start = False
            #     test_start_publisher.publish_once(test_start)
            # elif not test_start:
            #     test_start = True
            #     test_start_publisher.publish_once(test_start)
            # print(f"{T_total}")
            h_min = np.inf

            loop_start = time.perf_counter()



            obstacle_positions, obstacle_velocities, obstacle_accelerations = bridge.getObstacles()

            # print ("obstacle_positions:", obstacle_positions)
            # print ("type(obstacle_positions):", type(obstacle_positions))
            # print("size(obstacle_positions): ", len(obstacle_positions))
            cycles += 1

            nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time % T_total)
           
            out = ctrl.step(
                obs_pos=obstacle_positions,
                obs_vel=obstacle_velocities,
                obs_acc=obstacle_accelerations,
                nominal_q=nominal_q,
                nominal_Dq=nominal_Dq, 
                nominal_DDq=nominal_DDq
            )
            unfeasible_cnt = out["unfeasible_cnt"]
            q = out["q"]

            if cycles<5:
                print(f"q pln={nominal_q.T}\nq act={q.T}")
            dq = out["dq"]
            ddq = out["ddq"]
            trajectory_time = out["trajectory_time"]
            Dtrajectory_time = out["Dtrajectory_time"]

            elapsed = time.perf_counter() - loop_start
            ct_qp.append(elapsed)

            # --------------------------- INTEGRATION ----------------------------
            t += Tc

            if USE_BRIDGE and not stop_event.is_set():
                bridge.sendCommand(q)
            if not stop_event.is_set():
                joint_target_publisher.publish_once(nominal_q, nominal_Dq, nominal_DDq) # pyright: ignore[reportPossiblyUnboundVariable]
                hmin = out["h_min"]
                dmin = out["d_min"]
                trj_error = out["trajectory_error"]
                end_eff_pos = out["end_effector_pos"]
                end_eff_vel = out["end_effector_vel"]
                vr_min = out["vr_min"]
                vh_min = out["vh_min"]
                scaling = out["Dtrajectory_time"]
                cbf_out_publisher.publish_once(
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
                human_pos_publisher.publish_once(obstacle_positions)
            if not USE_BRIDGE and LOG_DATA:
                joint_state_publisher.publish_once(q, dq, ddq)
            # ----------------------------- TIMING -------------------------------
            elapsed = time.perf_counter() - loop_start
            if cycles>1:
                ct.append(elapsed)
                scaling_log.append(Dtrajectory_time)
                h_log.append(out["h_min"])
                trj_error_log.append(out["trajectory_error"])

            rest = Tc - elapsed
            if rest > 0:
                vizualization_string =f"h={out['h_min']:.2f}m  scale={out['Dtrajectory_time']:.3f}  err={out['trajectory_error']:.2f}"

                renderer.push_state(out["q"], out["Tbt_nominal"], out["obs_pos"], vizualization_string)
                elapsed = time.perf_counter() - loop_start
                rest = max(0.0,Tc - elapsed)
                time.sleep(rest)
            else:
                timeout_cycles+=1
        if not stop_event.is_set():
            test_start_publisher.publish_once(False) # pyright: ignore[reportPossiblyUnboundVariable]

    except KeyboardInterrupt:
        # request a graceful stop; loop condition will exit on next iteration
        stop_event.set()
# 
    finally:
        # bridge.shutdown()
        # time.sleep(0.1)
        # # 1) stop components that may be spinning their own executors (e.g., the bridge)
        # try:
        #     if 'bridge' in locals() and hasattr(bridge, 'shutdown'):
        #         bridge.shutdown()
        # except Exception:
        #     print("Error during bridge shutdown")
        # print(286)
       
        try:
            pub_utils.publish_test_start_once(False)
        except Exception as e:
            print(f"[shutdown] one-shot publish failed: {e}")
        print(304)
    # Call with your
    computation_times = np.array(ct)
    scaling_log = np.array(scaling_log)
    h_log = np.array(h_log)
    trj_error_log = np.array(trj_error_log)

    print(f"average scaling = {np.mean(scaling_log)}")

    #computation_times_others=computation_times-(computation_times_planner+computation_times_pin+computation_times_qp+computation_times_ssm)
    stats = {
        "computation_times": computation_times,
    }

    print(f"timeout cycles = {timeout_cycles} over {cycles}, percentage = {100.0*timeout_cycles/cycles}, average = {np.mean(computation_times)}")
    print(f"unfeasible cycles = {unfeasible_cnt} over {cycles}, percentage = {100.0*unfeasible_cnt/cycles}")

    print_stats_table(stats)
    _ = make_summary_figure(
        computation_times,
        h_log,
        trj_error_log,
        scaling_log,
    )


if __name__ == "__main__":
    main()
