# ur10_cbf_main.py
#
# UR10 kinematic simulation with CBF and Meshcat visualization.
# Controller implemented as a class with a .step() method.
# The QP is assembled via two methods:
#   - matrix_ensemble: non-CBF cost (P, b)
#   - cbf_ensemble: CBF inequality constraints (A, c)
import functools
import time
from typing import List

import meshcat.geometry as mgeom
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import math
import rclpy
from sharework import loadSharework

from example_cbf_optimal import _on_sigint_with_bridge
from interpolator import SegmentedSE3Trap
from visualization_daemon import VisualizationDaemon

import signal
import signal
import os
from datetime import datetime
import test_publish_utils as pub_utils
from PID_cbf_task_controller import UR10CBFController
import csv_publishers
import threading
USE_BRIDGE = False
LOG_DATA = False
log_path = "resullts/simulation/PID"
stop_event = threading.Event()
duration  = 150.0
def _on_sigint_with_bridge(bridge, signum, frame):
    stop_event.set()
    try:
        bridge.shutdown()
    except Exception:
        pass


def compute_ee_pose(q, model, data, ee_frame_id):
    """
    Compute forward kinematics of the end-effector for joint config q.
    Returns (position, rotation_matrix, SE3).
    """
    # Forward kinematics for all joints
    pin.forwardKinematics(model, data, q)
    # Update frame placements
    pin.updateFramePlacements(model, data)

    T_ee = data.oMf[ee_frame_id]  # SE3 from world (o) to frame (f=tool0)
    p = T_ee.translation          # 3D position
    R = T_ee.rotation             # 3x3 rotation matrix
    return p, R, T_ee

# ---------------------------- CONSTANTS --------------------------------------
C = 0.25   # [m]  minimum separation distance
Tr = 0.15  # [s]  controller-reaction time
a_s = 2.5  # [m/s²] robot decel/accel capability
Tc = 2e-3  # [s]   2 kHz control period
Dq_max: np.ndarray = np.pi * np.array([1, 1, 1, 1, 1, 1], dtype=np.float64) * np.pi*0.25

DDq_max: np.ndarray = np.pi * np.array([1, 1, 1, 1, 1, 1], dtype=np.float64) * np.pi * 5.0*0.2

def main():
    # --------------------------- MODEL & VISUALS -------------------------- #
    lap_count = 0
    on_target_count = 0
    prec_target = -1
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
    model = model_wrapper.model
    joint_ids = {}
    frame_ids = []
    for name in [ "ur10e_elbow_joint", "ur10e_wrist_3_joint",]:
        # --- Joint ID ---
        jid = model.getJointId(name)
        joint_ids[name] = jid

        # --- Frame ID (if a frame with that name exists) ---
        try:
            fid = model.getFrameId(name)
        except Exception:
            fid = None  # no frame with exactly that name
        frame_ids.append(fid)

    viz = MeshcatVisualizer(model, model_wrapper.collision_model, model_wrapper.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    # ------- BRIDGE SETUP -----------
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
            csv_path="skeleton_vectors/skeleton_vectors_14_NORMAL_TEST1.csv",
            Tworld_to_cam=T_wc,
            # slowdown_factor=0.1,
            slowdown_factor=1.0,
            t0 = 0.0

        )
        rclpy.init()
        first_joint_position = home



    # Obstacles (red spheres)
    tmp = np.array([-300, 0., 0.])
    obstacle_positions = [tmp.copy() for _ in range(18 * 5)]
    tmp = np.array([0, 0., 0.])
    obstacle_velocities = [tmp.copy() for _ in range(18 * 5)]
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

    renderer = VisualizationDaemon(viz)  # default 60 Hz

    # ------------------------ CONTROL INITIALISATION ---------------------- #
    data = model.createData()

    q = first_joint_position.copy()
    
    tool_frame_name = target_name

    # Gains (same as original)
    # wn =  59.187270296013395
    # xi = 0.21842533124715255
    gamma_default = 10 # CBF gain2.126488902627753
    wn =  169.88226052057638
    xi = 0.7423736875851532
    # gamma_default = 10 # CBF gain2.126488902627753

    Kp_tra = np.array([1.0, 1.0, 1.0]) * wn ** 2
    Kd_tra = np.array([1.0, 1.0, 1.0]) * 2.0 * xi * wn
    Kp_rot = np.array([1.0, 1.0, 1.0]) * wn ** 2
    Kd_rot = np.array([1.0, 1.0, 1.0]) * 2.0 * xi * wn

    # Controller instance (like 0 in your joint-space example)
    ctrl = UR10CBFController(
        model=model,
        tool_frame_name=tool_frame_name,
        frames_ids=frame_ids,
        Tc=Tc,
        Kp_tra=Kp_tra,
        Kd_tra=Kd_tra,
        Kp_rot=Kp_rot,
        Kd_rot=Kd_rot,
        gamma=gamma_default,
    )
    dq = np.zeros(model.nq)
    # We need initial frame pose for the planner
    pin.framesForwardKinematics(model, data, q)
    tool_frame_id = model.getFrameId(tool_frame_name)
    goal_pose_0 = data.oMf[tool_frame_id].copy()


    # 3) Split into linear and angular parts
    v_lin_max = 26.6586*0.1*0.055  # linear velocity [m/s]
    w_max = (44.1351 *0.1*0.055) # angular velocity [rad/s]

    a_lin_max = 650*0.1*0.1  # linear acceleration [m/s^2]
    alpha_max = 750 *0.1*0.1 # angular acceleration [rad/s^2]

    print(f"v_lin_max: {v_lin_max}")
    print(f"w_max: {w_max}")
    print(f"a_lin_max: {a_lin_max}")
    print(f"alpha_max: {alpha_max}")
   


    # BRING THE ROBOT AT HOME BEFORE STARTING THE TEST
    if USE_BRIDGE:
        start_ctrl = UR10CBFController(
        model=model.copy(),
        tool_frame_name=tool_frame_name,
        frames_ids=frame_ids,
        Tc=Tc,
        Kp_tra=Kp_tra,
        Kd_tra=Kd_tra,
        Kp_rot=Kp_rot,
        Kd_rot=Kd_rot,
        gamma=gamma_default,
    )
        start_planner = SegmentedSE3Trap(vlin_max=v_lin_max, vang_max=w_max,
                               alin_max=a_lin_max, aang_max=alpha_max)

        print(f"Bringing robot to home position from {q.T} to {home.T}")
        p, R, T_ee = compute_ee_pose(q, model, data, tool_frame_id)
        start_planner.addWayPoint(T_ee)
        p, R, T_ee = compute_ee_pose(home, model, data, tool_frame_id)
        start_planner.addWayPoint(T_ee)
        t_initial = 0.0

        start_time = start_planner.computeTime()
        print(f"Bringing robot to home position, total time: {start_time}")
        time.sleep(1.0)
        start_ctrl.reset_state(q)
        # test_start = True
        while np.linalg.norm(home-bridge.getPositions()) > 0.001:
            loop_start = time.perf_counter()
            obstacle_positions, obstacle_velocities, obstacle_accelerations = bridge.getObstacles()

            goal_pose, nominal_twist_goal, nominal_goal_dtwist = start_planner.getMotionLaw(t_initial)
             # Scale if you ever implement time-scaling; currently D=1, DD=0
            Dtrajectory_time = 1.0
            DDtrajectory_time = 0.0
            twist_goal = nominal_twist_goal * Dtrajectory_time
            goal_dtwist = (
                nominal_goal_dtwist * Dtrajectory_time ** 2.0
                + nominal_twist_goal * DDtrajectory_time
            )
            out = start_ctrl.step(
                t=t_initial,
                goal_pose=goal_pose,
                twist_goal=twist_goal,
                goal_dtwist=goal_dtwist,
                obstacle_positions=obstacle_positions,
                obstacle_velocities=obstacle_velocities,
                obstacle_accelerations=obstacle_accelerations,
                cbf_enabled=True,
            )

            q = out["q"]
            bridge.sendCommand(q)
            
            # --------------------------- INTEGRATION ----------------------------
            t_initial += Tc

            elapsed = time.perf_counter() - loop_start
            
            rest = Tc - elapsed
            if rest > 0:
                rest = max(0.0,Tc - elapsed)
                time.sleep(rest)
        q = home.copy()

    print(f"Starting main loop from q: {q.T}")
    q10 = np.array([31.0, -78.0, 115.0, -127.0, 86.0, -32.0]) * np.pi / 180.0
    q20 = np.array([31.0, -83.0, 98.0, -110.0, 86.0, -32.0]) * np.pi / 180.0
    q22 = np.array([40.0, -126.0, 141.0, -100.0, 86.0, 45.0]) * np.pi / 180.0
    q25 = np.array([130.0, -100.0, 125.0, -115.0, 94.0, -20.0]) * np.pi / 180.0
    q30 = np.array([136.0, -60.0, 90.0, -122.0, 90.0, 45.0]) * np.pi / 180.0
    q40 = np.array([134.0, -65.0, 70.0, -90.0, 90.0, 45.0]) * np.pi / 180.0

    configs = {
        "q": q,
        "q10": q10,
        "q20": q20,
        "q22": q22,
        "q25": q25,
        "q30": q30,
        "q40": q40,
    }
    ordered_configs = []

    ordered_configs.extend(["q", "q10", "q20", "q10", "q22", "q25", "q30", "q40", "q30", "q"])
    cartesian_configs = {
        "q": 0.0,
        "q10": 0.0,
        "q20": 0.0,
        "q22": 0.0,
        "q25": 0.0,
        "q30": 0.0,
        "q40": 0.0,
    }

     # -------------------------- Trajectory planner ------------------------ #
    planner = SegmentedSE3Trap(vlin_max=v_lin_max*2.4, vang_max=w_max*2.4,
                               alin_max=a_lin_max*1.1, aang_max=alpha_max*1.1)

    for name in ordered_configs:
        p, R, T_ee = compute_ee_pose(configs[name], model, data, tool_frame_id)
        planner.addWayPoint(T_ee)

    for name in cartesian_configs:
        p, R, T_ee = compute_ee_pose(configs[name], model, data, tool_frame_id)
        cartesian_configs[name] = p.tolist()
    # for i in range(len(ordered_configs)):
    #     p, R, T_ee = compute_ee_pose(home, model, data, tool_frame_id)
    #     planner.addWayPoint(T_ee)
    # print("Building trajectory...")
    # print("Q: ", q)
    # p, R, T_ee = compute_ee_pose(q, model, data, tool_frame_id)
    # planner.addWayPoint(T_ee)
    # p, R, T_ee = compute_ee_pose(home, model, data, tool_frame_id)
    # planner.addWayPoint(T_ee)
    # p, R, T_ee = compute_ee_pose(q, model, data, tool_frame_id)
    # planner.addWayPoint(T_ee)


    T_total = planner.computeTime()


    renderer.publishPath(planner.publishPath())
    print(f"Total time = {T_total:.3f} s")
    # ------------------------ PUBLISHER TARGETS  SETUP-----------------------------------
    if LOG_DATA:
        if USE_BRIDGE:
            joint_target_publisher = pub_utils.JointTargetPublisher(
                topic='joint_target',
                joint_names=["target_x","target_y","target_z"],
                frame_id='world'
            )

            test_start_publisher = pub_utils.TestStartPublisher(
                topic='test_start'
            )
            cbf_out_publisher = pub_utils.DoubleArrayPublisher(
                topic='cbf_output',
                node_name='cbf_output_publisher', )
            # dim = 10)
            human_pos_publisher = pub_utils.DoubleArrayPublisher(
                topic='human_pos_keypoints',
                node_name='human_pos_publisher', )
        else:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_path = log_path + "/" + str(now)
            # now = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
            print(test_path)
            os.makedirs(test_path, exist_ok=True)
            joint_target_publisher = csv_publishers.JointTargetCsvPublisher(
                csv_path=test_path + "/reference_trajectory.csv",
                column_names="time,x,vel_x,acc_x,y,vel_y,acc_y,z,vel_z,acc_z",
                joint_names=["x","y","z"],
            )
            # JOINT STATE PUBLISHER ONLY FOR CSV LOGGING
            joint_state_publisher = csv_publishers.JointTargetCsvPublisher(
                csv_path=test_path + "/joint_states.csv",
                column_names="time,joint_0_pos,joint_0_vel,joint_0_acceleration,joint_1_pos,joint_1_vel,joint_1_acceleration,joint_2_pos,joint_2_vel,joint_2_acceleration,joint_3_pos,joint_3_vel,joint_3_acceleration,joint_4_pos,joint_4_vel,joint_4_acceleration,joint_5_pos,joint_5_vel,joint_5_acceleration",
                joint_names=UR10E_JOINTS,
            )

            test_start_publisher = csv_publishers.TestStartCsvPublisher(
                csv_path=test_path + "/TEST_START.csv",
                column_names="time,val"
            )
            cbf_out_publisher = csv_publishers.DoubleArrayCsvPublisher(
                csv_path=test_path + "/cbf_results.csv",
                column_names="time,h_min,d_min,trajectory_error,pos_ee_x,pos_ee_y,pos_ee_z,vel_ee_x,vel_ee_y,vel_ee_z,v_r_min,v_h_min")
            # dim = 10)
            human_pos_publisher = csv_publishers.DoubleArrayCsvPublisher(
                csv_path=test_path + "/human_positions.csv",
                column_names="time,human_keypoint_0_x,human_keypoint_0_y,human_keypoint_0_z,human_keypoint_1_x,human_keypoint_1_y,human_keypoint_1_z,human_keypoint_2_x,human_keypoint_2_y,human_keypoint_2_z,human_keypoint_3_x,human_keypoint_3_y,human_keypoint_3_z,human_keypoint_4_x,human_keypoint_4_y,human_keypoint_4_z,human_keypoint_5_x,human_keypoint_5_y,human_keypoint_5_z,human_keypoint_6_x,human_keypoint_6_y,human_keypoint_6_z,human_keypoint_7_x,human_keypoint_7_y,human_keypoint_7_z,human_keypoint_8_x,human_keypoint_8_y,human_keypoint_8_z,human_keypoint_9_x,human_keypoint_9_y,human_keypoint_9_z,human_keypoint_10_x,human_keypoint_10_y,human_keypoint_10_z,human_keypoint_11_x,human_keypoint_11_y,human_keypoint_11_z,human_keypoint_12_x,human_keypoint_12_y,human_keypoint_12_z,human_keypoint_13_x,human_keypoint_13_y,human_keypoint_13_z,human_keypoint_14_x,human_keypoint_14_y,human_keypoint_14_z,human_keypoint_15_x,human_keypoint_15_y,human_keypoint_15_z,human_keypoint_16_x,human_keypoint_16_y,human_keypoint_16_z,human_keypoint_17_x,human_keypoint_17_y,human_keypoint_17_z"
            )

    # ------------------------------ MAIN LOOP ---------------------------- #


    if LOG_DATA and USE_BRIDGE:
        test_start_publisher.publish_once(True) # pyright: ignore[reportPossiblyUnboundVariable]
    try:
        ctrl.reset_state(q)
        t = 0.0
        trajectory_time = 0.0
        Dtrajectory_time = 1.0
        DDtrajectory_time = 0.0
        while t < duration and not stop_event.is_set():
            loop_start = time.perf_counter()

            if T_total >0.0:
                goal_pose, nominal_twist_goal, nominal_goal_dtwist = planner.getMotionLaw(
                    trajectory_time % T_total
                )
            else:
                goal_pose, nominal_twist_goal, nominal_goal_dtwist = planner.getMotionLaw(
                trajectory_time
                )
            
            if USE_BRIDGE:
                obstacle_positions, obstacle_velocities, obstacle_accelerations = bridge.getObstacles()
            else:
                obstacle_positions, obstacle_velocities, obstacle_accelerations = bridge.getObstacles(elapsed = t)
                # print("ELAPSED: ", t)
            # elapsed = time.perf_counter() - loop_start
            # print("elapsed time: ", elapsed)

            # Scale if you ever implement time-scaling; currently D=1, DD=0
            twist_goal = nominal_twist_goal * Dtrajectory_time
            goal_dtwist = (
                nominal_goal_dtwist * Dtrajectory_time ** 2.0
                + nominal_twist_goal * DDtrajectory_time
            )

            # # CBF toggle (10 s on / 10 s off)
            # cbf_enabled = (t % 40.0) < 20.0

            # --------- Controller step (this is the key new API) ------------- #
            out = ctrl.step(
                t=t,
                goal_pose=goal_pose,
                twist_goal=twist_goal,
                goal_dtwist=goal_dtwist,
                obstacle_positions=obstacle_positions,
                obstacle_velocities=obstacle_velocities,
                obstacle_accelerations=obstacle_accelerations,
            )
            # elapsed = time.perf_counter() - loop_start
            # print("elapsed time: ", elapsed)

            q = out["q"]
            dq = out["dq"]
            ddq = out["ddq"]
            h_min = out["h_min"]
            # Print q and dq for debugging
            # print(f"q: {q} \n dq: {dq}")

            # --------------------------- TIMING & VISUALS ------------------- #
            t += Tc
            trajectory_time += Dtrajectory_time * Tc + 0.5 * DDtrajectory_time * Tc ** 2.0
            Dtrajectory_time += DDtrajectory_time * Tc
            if 0 < (trajectory_time % T_total) < Tc:
                lap_count += 1
                print("LAP ADDED")
            if USE_BRIDGE:
                # print(f"Sending command: {q}")
                bridge.sendCommand(q)
            end_eff_pos = out["end_effector_pos"]

            if not stop_event.is_set() and LOG_DATA:
                nom_x, nom_y, nom_z = goal_pose.translation.tolist()
                joint_target_publisher.publish_once(t, [nom_x, nom_y, nom_z], twist_goal[0:3], goal_dtwist[0:3])
                hmin = out["h_min"]
                dmin = out["d_min"]
                trj_error = out["trajectory_error"]
                end_eff_vel = out["end_effector_vel"]
                vr_min = out["vr_min"]
                vh_min = out["vh_min"]
                cbf_out_publisher.publish_once( t,
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
                    ]
                )  # pyright: ignore[reportPossiblyUnboundVariable]
                human_pos_publisher.publish_once(t, obstacle_positions)
            if not USE_BRIDGE and LOG_DATA:
                joint_state_publisher.publish_once(t, q, dq, ddq)

            for i in range(len(cartesian_configs.values())):
                q_wp = list(cartesian_configs.values())[i]
                if np.linalg.norm(q_wp - end_eff_pos) < 2e-03 and prec_target != i:
                    on_target_count += 1
                    prec_target = i
                    print("TARGET REACHED")
                    break
            elapsed = time.perf_counter() - loop_start
            rest = Tc - elapsed

            vizualization_string = f"h = {h_min:.2f} m, err={out['trajectory_error']:.2f}"
            if rest > 0:
                time.sleep(0.0001)
                # renderer.push_state(
                #     q,
                #     goal_pose,
                #     obstacle_positions,
                #     vizualization_string,
                # )
                # # account for visualization time as well
                # elapsed = time.perf_counter() - loop_start
                # rest = max(0.0, Tc - elapsed)
                # time.sleep(rest)
                pass
            else:
                print(f"TIMEOUT, elapsed:{elapsed:.4f}")
        print ("FINE CICLO")
        if not stop_event.is_set() and LOG_DATA:
            test_start_publisher.publish_once(False) # pyright: ignore[reportPossiblyUnboundVariable]

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        stop_event.set()


    finally:
       
        try:
            pub_utils.publish_test_start_once(False)
        except Exception as e:
            print(f"[shutdown] one-shot publish failed: {e}")
    n_wp = 9
    print(f"LAP COUNT: {lap_count}")
    print("on target count: ", on_target_count)
    print(((trajectory_time % T_total) / T_total))
    print(f"WAYPOINTS REACHING PERCENTAGE: {on_target_count / (n_wp * ((lap_count) + ((trajectory_time % T_total) / T_total)))}")

if __name__ == "__main__":
    main()
