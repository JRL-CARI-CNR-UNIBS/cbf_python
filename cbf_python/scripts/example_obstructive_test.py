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
import random
import time

import meshcat.geometry as mgeom

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from scripts.util.joint_interpolator import SegmentedJointTrap
from scripts.util.visualization_daemon import VisualizationDaemon
from sharework import loadSharework

from Controller.optimal_cbf_task_controller import BCFOptimalController, ControllerConfig

from datetime import datetime
import rclpy

import threading
from scripts.util import csv_publishers, test_publish_utils as pub_utils
from scripts.util.reference_xyz_trajectory import generate_cartesian_trajectory
from scripts.util.test_utils import generate_obs_state, compute_ee_pose, generate_velocity

import csv
from scripts.util.gaussian_process_util import generate_d_value, generate_obs_state_h_fixed, compute_required_d, generate_target_h, read_config_data_from_csv
from scripts.util.mean_visualizer import StochasticCBFVisualizer
set_ID = "0"
duration = 150.0

SHOW_DATA = True
LOG_DATA = False
SAVE_DATA = False
parameters_type = "0"
stop_event = threading.Event()

h_cfg = "article"
v_cfg = "article"
# h_cfg = 1
# v_cfg = 1



h_mean_ref = -0.1
v_ref = 1
spawn_freq = 10
h_std_dev = 0.15
test_name= f"TEST_OBSTRUCTIVE_h_mean_{h_mean_ref:.2f}_v_mean_{v_ref:.2f}_paper_par"
# d_objective = 0.1
d_objective = generate_d_value(h_mean_ref, 0.1)
def _on_sigint_with_bridge():
    stop_event.set()



# signal.signal(signal.SIGINT, _on_sigint_with_bridge)

def main():
    # --------------------------- MODEL & VISUALS ---------------------------------
    log_path = "../resullts/simulation/scaling"
    # rclpy.init()

    visualizer = StochasticCBFVisualizer()
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
    Tc = 2e-3
    cfg = ControllerConfig(Tc=Tc)
    # PAPER PARAMETERS
    # cfg.lambda_pos =  1083.9977322239226
    # cfg.lambda_vel = 0.019463569108586626
    # cfg.lambda_scaling =   88.92080107598409
    # cfg.lambda_acc =  9.684370933446392e-08
    # delta = 4.427823857718463
    # cfg.gamma =   9.651586852673113

    # df = pd.read_csv(params_filename)
    #
    # cfg.lambda_pos = float(df.loc[df["ID"] == set_ID, f"lambda_{parameters_type}_pos"].values[0])
    # cfg.lambda_vel = float(df.loc[df["ID"] == set_ID, f"lambda_{parameters_type}_vel"].values[0])
    # cfg.lambda_acc = float(df.loc[df["ID"] == set_ID, f"lambda_{parameters_type}_acc"].values[0])
    # cfg.lambda_scaling = float(df.loc[df["ID"] == set_ID, f"lambda_{parameters_type}_scaling"].values[0])
    # cfg.gamma = float(df.loc[df["ID"] == set_ID, f"gamma_{parameters_type}"].values[0])
    # delta = float(df.loc[df["ID"] == set_ID, f"delta_{parameters_type}_deg"].values[0])

    delta = 4.5

    read_config_data_from_csv(cfg,h_mean=h_cfg, v_mean=v_cfg, filename="../log_best_trials.csv")
    cfg.delta_q_max[0:2] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta)
    cfg.delta_q_max[2:4] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 2
    cfg.delta_q_max[4:6] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 4
    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True, keypoint_to_log=-1)
    print(cfg)
    target_name = "ur10e_wrist_3_joint"
    idx = UR10E_JOINTS.index(target_name)

    rclpy.init()
    first_joint_position = home
    # ------------------------ PUBLISHER TARGETS  SETUP-----------------------------------
    if LOG_DATA:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_path = log_path + "/" + str(now)
        # now = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
        print(test_path)
        os.makedirs(test_path, exist_ok=True)
        joint_target_publisher = csv_publishers.JointTargetCsvPublisher(
            csv_path=test_path + "/reference_trajectory.csv",
            column_names="time,target_joint_0_pos,target_joint_0_vel,target_joint_0_acceleration,target_joint_1_pos,target_joint_1_vel,target_joint_1_acceleration,target_joint_2_pos,target_joint_2_vel,target_joint_2_acceleration,target_joint_3_pos,target_joint_3_vel,target_joint_3_acceleration,target_joint_4_pos,target_joint_4_vel,target_joint_4_acceleration,target_joint_5_pos,target_joint_5_vel,target_joint_5_acceleration",
            joint_names=UR10E_JOINTS,
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
            column_names="time,h_min,d_min,trajectory_error,pos_ee_x,pos_ee_y,pos_ee_z,vel_ee_x,vel_ee_y,vel_ee_z,v_r_min,v_h_min,scaling")
        # dim = 10)
        human_pos_publisher = csv_publishers.DoubleArrayCsvPublisher(
            csv_path=test_path + "/human_positions.csv",
            column_names="time,human_keypoint_0_x,human_keypoint_0_y,human_keypoint_0_z"
        )

    model = model_wrapper.model
    viz = MeshcatVisualizer(model, model_wrapper.collision_model, model_wrapper.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    tmp = np.array([-300, 0., 0.])
    obstacle_positions = [tmp.copy() for _ in range(18 * 5)]
    tmp = np.array([0, 0., 0.])
    obstacle_velocities = [tmp.copy() for _ in range(18 * 5)]

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
    q10 = np.array([31.0, -78.0, 115.0, -127.0, 86.0, -32.0]) * np.pi / 180.0
    q20 = np.array([31.0, -83.0, 98.0, -110.0, 86.0, -32.0]) * np.pi / 180.0
    q22 = np.array([40.0, -126.0, 141.0, -100.0, 86.0, 45.0]) * np.pi / 180.0
    q25 = np.array([130.0, -100.0, 125.0, -115.0, 94.0, -20.0]) * np.pi / 180.0
    q30 = np.array([136.0, -60.0, 90.0, -122.0, 90.0, 45.0]) * np.pi / 180.0
    q40 = np.array([134.0, -65.0, 70.0, -90.0, 90.0, 45.0]) * np.pi / 180.0
    cfg.Dq_max = cfg.Dq_max * 0.25
    cfg.DDq_max = cfg.DDq_max * 0.2
    planner = SegmentedJointTrap(Dq_max=cfg.Dq_max * 0.25, DDq_max=cfg.DDq_max * 0.25)

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
    n_wp = 10
    configs = {
        "q": q,
        "q10": q10,
        "q20": q20,
        "q22": q22,
        "q25": q25,
        "q30": q30,
        "q40": q40,
    }
    cartesian_configs = {
        "q": 0.0,
        "q10": 0.0,
        "q20": 0.0,
        "q22": 0.0,
        "q25": 0.0,
        "q30": 0.0,
        "q40": 0.0,
    }
    tool_frame_name = "ur10e_wrist_3_joint"
    tool_frame_id = model.getFrameId(tool_frame_name)
    data = model.createData()
    for name in cartesian_configs:
        p, R, T_ee = compute_ee_pose(configs[name], model, data, tool_frame_id)
        cartesian_configs[name] = p.tolist()

    T_total = planner.computeTime()
    print(f"Total time: {T_total}")
    min_dist = []
    renderer.publishPath(planner.publishPath())

    ct, ct_qp, ct_ssm, ct_planner, ct_pin, h_log, trj_error_log, scaling_log = [], [], [], [], [], [], [], []

    lap_count = 0
    on_target_count = 0
    # ------------------------------ MAIN LOOP -------------------- ----------------
    prec_target = -1
    enable_lap_count = True
    if LOG_DATA:
        test_start_publisher.publish_once(True)  # pyright: ignore[reportPossiblyUnboundVariable]
    unfeasible_cnt = 0
    try:

        t = 0.0
        trajectory_time = 0.0
        timeout_cycles = cycles = 0
        violations = sum_scale = trajectory_error_sum = 0

        ctrl.reset_state(q)
        # test_start = True
        obstacle_positions = np.zeros(3)
        obstacle_velocities = np.zeros(3)
        obstacle_accelerations = np.array([20.0,20.0,20.0])*0.0
        ee_pos = np.zeros(3)
        ee_vel = np.zeros(3)
        count_move = 0
        Dtrajectory_time = 1.0
        low_scale_count = 0
        scaling_threshold = 0.5
        consecutive_low_scale_cycles = 0
        enable_spawn = True
        obstacle_accelerations = obstacle_accelerations.reshape(1, 3)
        vr_min = -0.1
        # ctrl.frames_ids = [ctrl.tool_frame_id]
        while t < duration and not stop_event.is_set():
            loop_start = time.perf_counter()
            nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time % T_total)

            # obstacle_positions, obstacle_velocities, enable_spawn, count_move = generate_obs_state(obstacle_positions, obstacle_velocities, cycles, enable_spawn, planner, trajectory_time, T_total, model, data, tool_frame_id, ee_pos, Dtrajectory_time, count_move)
            h_objective = generate_target_h(h_mean_ref, h_std_dev)
            d_objective = compute_required_d(h_objective, vr_min, v_ref, np.linalg.norm(obstacle_accelerations) )
            obstacle_positions, obstacle_velocities, enable_spawn, count_move = generate_obs_state_h_fixed(obstacle_positions, obstacle_velocities, cycles, enable_spawn, ctrl.model, ctrl.data, tool_frame_id, ee_pos, Dtrajectory_time, count_move, d_objective, v_ref, spawn_freq, ee_vel)#nominal_q, nominal_Dq, nominal_DDq)
            #     # print(obstacle_positions)
            #     print(f"TYPE OF OBSTACLE POSITIONS: {type(obstacle_positions)}")
            #     print(f"TYPE OF OBSTACLE Velocities: {type(obstacle_velocities)}")

                # print(f"SIZE OF OBS POSITIONS: {obstacle_positions.shape}")
            cycles += 1


            out = ctrl.step(
                obs_pos=obstacle_positions,
                obs_vel=obstacle_velocities,
                obs_acc=obstacle_accelerations,
                nominal_q=nominal_q,
                nominal_Dq=nominal_Dq,
                nominal_DDq=nominal_DDq
            )
            ee_pos = out["end_effector_pos"]
            ee_vel = out["end_effector_vel"]
            # print(np.linalg.norm(ee_vel))
            unfeasible_string = out["unfeasible_cnt"]
            q = out["q"]

            if cycles < 5:
                print(f"q pln={nominal_q.T}\nq act={q.T}")
            dq = out["dq"]
            ddq = out["ddq"]
            trajectory_time = out["trajectory_time"]
            Dtrajectory_time = out["Dtrajectory_time"]
            if (trajectory_time % T_total) < Tc:
                if enable_lap_count:
                    lap_count += 1
                    prec_target = -1
                    enable_lap_count = False
            else:
                enable_lap_count = True
                # print(f"actual lap: {int(trajectory_time % T_total)}")
            elapsed = time.perf_counter() - loop_start
            ct_qp.append(elapsed)

            # --------------------------- INTEGRATION ----------------------------
            t += Tc
            end_eff_pos = out["end_effector_pos"]
            # print(t)
            vr_min = out["vr_min"]

            if not stop_event.is_set() and LOG_DATA:
                joint_target_publisher.publish_once(t, nominal_q, nominal_Dq, nominal_DDq)  # pyright: ignore[reportPossiblyUnboundVariable]
                hmin = out["h_min"]
                dmin = out["d_min"]
                trj_error = out["trajectory_error"]
                end_eff_vel = out["end_effector_vel"]
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
                )  # pyright: ignore[reportPossiblyUnboundVariable]
                human_pos_publisher.publish_once(t, obstacle_positions)
            if LOG_DATA:
                joint_state_publisher.publish_once(t, q, dq, ddq)
            # ----------------------------- TIMING -------------------------------
            dist = []
            for i in range(len(cartesian_configs.values())):
                q_wp = list(cartesian_configs.values())[i]
                dist.append(np.linalg.norm(q_wp - end_eff_pos))
                if np.linalg.norm(q_wp - end_eff_pos) < 2e-03 and prec_target != i:
                    on_target_count += 1
                    prec_target = i
                    break
            if np.min(dist) > 0.0:
                min_dist.append(np.min(dist))
            if cycles > 1:
                ct.append(elapsed)
                scaling_log.append(Dtrajectory_time)
                h_log.append(out["h_min"])
                trj_error_log.append(out["trajectory_error"])
            if (out["h_min"] < (h_objective + 1.5 * h_std_dev) and out["h_min"] > (h_objective - 1.5 * h_std_dev)) or SAVE_DATA:
                if out["h_min"] < 0 and out["vr_min"] < -1e-3:
                    violations += 1
                sum_scale += out["Dtrajectory_time"]
                trajectory_error_sum += out["trajectory_error"]

                if out["Dtrajectory_time"] < scaling_threshold:
                    low_scale_count += 1

                visualizer.update_vectors(out["h_min"], out["d_min"], out["vr_min"] - out["vh_min"], t,)
            elapsed = time.perf_counter() - loop_start

            rest = Tc - elapsed
            if rest > 0:
                if SHOW_DATA:
                    vizualization_string = f"h={out['h_min']:.2f}m  scale={out['Dtrajectory_time']:.3f}  err={out['trajectory_error']:.2f} ctrl_state:{unfeasible_string}"

                    renderer.push_state(out["q"], out["Tbt_nominal"], obstacle_positions, obstacle_velocities,vizualization_string)
                    elapsed = time.perf_counter() - loop_start
                    rest = max(0.0,Tc - elapsed)
                    time.sleep(rest)
                else:
                    time.sleep(0.0001)
            else:
                timeout_cycles += 1
            if unfeasible_string != "FEASIBLE":
                unfeasible_cnt += 1
        if not stop_event.is_set() and LOG_DATA:
            test_start_publisher.publish_once(False)  # pyright: ignore[reportPossiblyUnboundVariable]

    except KeyboardInterrupt:
        # request a graceful stop; loop condition will exit on next iteration
        stop_event.set()
    #
    finally:
        try:
            pub_utils.publish_test_start_once(False)
        except Exception as e:
            print(f"[shutdown] one-shot publish failed: {e}")

    computation_times = np.array(ct)
    scaling_log = np.array(scaling_log)



    print(f"average scaling = {np.mean(scaling_log)}")

    # computation_times_others=computation_times-(computation_times_planner+computation_times_pin+computation_times_qp+computation_times_ssm)
    stats = {
        "computation_times": computation_times,
    }
    lap_count = lap_count + ((trajectory_time % T_total) / T_total)
    on_target_rate = on_target_count / (n_wp * ((lap_count) + ((trajectory_time % T_total) / T_total)))
    viol_rate = violations / len(visualizer.h_vec)
    mean_scale = sum_scale / len(visualizer.h_vec)
    mean_trajectory_error = trajectory_error_sum / len(visualizer.h_vec)
    low_scale_rate = low_scale_count / len(visualizer.h_vec)

    print(
        f"timeout cycles = {timeout_cycles} over {cycles}, percentage = {100.0 * timeout_cycles / cycles}, average = {np.mean(computation_times)}")
    print(f"unfeasible cycles = {unfeasible_cnt} over {cycles}, percentage = {100.0 * unfeasible_cnt / cycles} %")
    print(f"LAP COUNT: {lap_count}")
    print("on target count: ", on_target_count)
    print(((trajectory_time % T_total) / T_total))
    print(f"WAYPOINTS REACHING PERCENTAGE: {on_target_rate * 100.0} %")
    print(f"VIOLATION RATE: {viol_rate}")
    print(f"MEAN SCALING: {mean_scale}")
    print(f"MEAN TRAJECTORY ERROR: {mean_trajectory_error}")
    print(f"LOW SCALE RATE: {low_scale_rate}")
    # print(f"D OBJECTIVE: {d_objective}")
    print((f"V REF: {v_ref}"))
    print(f"Cicli contati: {len(visualizer.h_vec)}, cicli totali: {cycles}")
    print(f"Percentuale cicli utili: {len(visualizer.h_vec)/cycles}")
    visualizer.compute_mean_cov(True)
    # print_stats_table(stats)
    # _ = make_summary_figure(
    #     computation_times,
    #     h_log,
    #     trj_error_log,
    #     scaling_log,
    # )
    folder_name = ""
    # CREATING CARTESIAN REFERENCE CSV FILE
    if LOG_DATA:
        folder_name = test_path
        generate_cartesian_trajectory(folder_name + "/")
    #
    # # SAVING RESULTS
    if SAVE_DATA:
        print("SAVING RESULTS")
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
            'low_scale_rate',
        ]

        # I dati da salvare (calcolati come nel tuo esempio)
        row_data = {
            "test_type": test_name,
            "lambda_pos": cfg.lambda_pos,
            "lambda_vel": cfg.lambda_vel,
            "lambda_scaling": cfg.lambda_scaling,
            "lambda_acc": cfg.lambda_acc,
            "delta": delta,
            "gamma": cfg.gamma,
            'on_target_rate': on_target_rate,
            'lap_count': lap_count,
            'viol_rate': viol_rate,
            'mean_scale': mean_scale,
            'mean_trajectory_error': mean_trajectory_error,
            'low_scale_rate' : low_scale_rate
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
