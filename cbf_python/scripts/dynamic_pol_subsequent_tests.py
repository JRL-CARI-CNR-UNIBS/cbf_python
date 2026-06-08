# version of example_dynamic_params that performs more tests in succession
import os
import csv
import time


import numpy as np
import pinocchio as pin

from scripts.util.joint_interpolator import SegmentedJointTrap
from sharework import loadSharework
from scripts.util.gaussian_process_util import generate_obs_state_h_fixed, compute_required_d, generate_target_h, \
    read_poly_config_data_from_csv, read_config_data_from_csv
import matplotlib.pyplot as plt


import rclpy

import threading

from Controller.dynamic_params_controllers import (PolynomialControllerConfig, PolynomialOptimalController, )
from Controller.optimal_cbf_task_controller import BCFOptimalController, ControllerConfig
from scripts.util.test_utils import compute_ee_pose, plan_path
from scripts.util.mean_visualizer import StochasticCBFVisualizer
from scripts.util.bcf_utils import plot_lambdas

stop_event = threading.Event()


# params_filename = "din_par.csv" #FILE WITH PARAMETERS
params_filename = "../dynamics_par_multicase_no_jump_h_mixed_top_10.csv"  # FILE WITH PARAMETERS
trial_name = "dynamic_params"
SAVE_DATA = True
PLOT_MEAN = True
PLOT_LAMBDAS = False
def _on_sigint_with_bridge(bridge, signum, frame):
    stop_event.set()
    try:
        bridge.shutdown()
    except Exception:
        pass

def run_experiment(test_type = "O",
                    duration = 15000.0,
                    h_mean_ref = -0.10,
                    h_std_dev = 0.2,

                    v_ref = 1.0,
                    spawn_freq = 10,

                    test_name = "GENERAL_TEST_h_high",
                    test_code = "",
                   controller_type = 0,
                   h_cfg = 0,
                   v_cfg = 0):
    # --------------------------- MODEL & VISUALS ---------------------------------

    print(f"test type: {test_type}")
    print(f"duration: {duration}")
    print(f"h_mean_ref: {h_mean_ref}")
    print(f"h_std_dev: {h_std_dev}")
    print(f"v_ref: {v_ref}")
    print(f"spawn_freq: {spawn_freq}")
    print(f"test_name: {test_name}")
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
    delta = 4.5
    if controller_type == 0:
        cfg = ControllerConfig(Tc=Tc)

        read_config_data_from_csv(cfg, h_mean=h_cfg, v_mean=v_cfg, filename="../log_best_trials.csv")
        cfg.delta_q_max[0:2] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta)
        cfg.delta_q_max[2:4] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 2
        cfg.delta_q_max[4:6] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 4
        print(cfg)

        ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True, keypoint_to_log=-1)
    if controller_type == 1:
        cfg = PolynomialControllerConfig(Tc=Tc)
        # PAPER PARAMETERS

        cfg.h_t = 1.0

        read_poly_config_data_from_csv(cfg=cfg, filename=params_filename, trial_name=trial_name)

        delta = 4.5

        cfg.delta_q_max[0:2] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta)
        cfg.delta_q_max[2:4] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 2
        cfg.delta_q_max[4:6] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 4
        print(f"config: {cfg}")
        ctrl = PolynomialOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True, keypoint_to_log=-1)
    gamma_list = []
    lambda_pos_list = []
    lambda_vel_list = []
    lambda_acc_list = []
    lambda_scaling_list = []
    t_list = []
    target_name = "ur10e_wrist_3_joint"
    idx = UR10E_JOINTS.index(target_name)

    from Command_bridge.fake_command_bridge import FakeCommandBridge
    # Build camera pose from your INITI snippet
    quat = pin.Quaternion(0.83, 0.185, 0.513, 0.12)
    quat.normalize()

    R = quat.toRotationMatrix()

    T_wc = pin.SE3(R, np.array([-0.094, -0.93, 2.309]))

    # csv_path= "/home/nyquist/projects/cells_ws/src/zed_skeleton_kinematics/csv_files/skeleton_vectors_23.csv"
    csv_path = "../skeleton_vectors/skeleton_vectors_23.csv"
    bridge = FakeCommandBridge(
        UR10E_JOINTS,
        csv_path=csv_path,
        Tworld_to_cam=T_wc,
        # slowdown_factor=0.1,
        slowdown_factor=1.0,
        t0=0.0

    )
    first_joint_position = home
    # ------------------------ PUBLISHER TARGETS  SETUP----------------------------------
    model = model_wrapper.model

    tmp = np.array([-300, 0., 0.])
    obstacle_positions = [tmp.copy() for _ in range(18 * 5)]
    tmp = np.array([0, 0., 0.])
    obstacle_velocities = [tmp.copy() for _ in range(18 * 5)]
    obstacle_accelerations = np.array([20.0, 20.0, 20.0]) * 0.0
    obstacle_accelerations = obstacle_accelerations.reshape(1, 3)

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
    print("Computing trajectory...")
        # 2 · add way‑points -------------------------------------------
    plan_path(planner, q)
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

    ct, ct_qp, ct_ssm, ct_planner, ct_pin, h_log, trj_error_log, scaling_log = [], [], [], [], [], [], [], []

    lap_count = 0
    on_target_count = 0
    # ------------------------------ MAIN LOOP -------------------- ----------------
    prec_target = -1
    enable_lap_count = True

    unfeasible_cnt = 0
    low_scale_count = 0
    scaling_threshold = 0.5
    visualizer = StochasticCBFVisualizer()
    try:

        t = 0.0
        trajectory_time = 0.0
        timeout_cycles = cycles = 0
        violations = sum_scale = trajectory_error_sum = 0

        count_move = 0
        end_eff_pos = np.zeros(3)
        Dtrajectory_time = 1.0
        ctrl.reset_state(q)
        # test_start = True
        enable_spawn = True

        if test_type == "O":
            obstacle_positions = np.zeros(3)
            obstacle_velocities = np.zeros(3)
            obstacle_accelerations = np.array([20.0, 20.0, 20.0]) * 0.0
            obstacle_accelerations = obstacle_accelerations.reshape(1, 3)
            ee_vel = np.zeros(3)
            vr_min = 0.0

        while t < duration and not stop_event.is_set():

            loop_start = time.perf_counter()
            nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time % T_total)


            if test_type == "O":
                h_objective = generate_target_h(h_mean_ref, h_std_dev)
                d_objective = compute_required_d(h_objective, vr_min, v_ref, np.linalg.norm(obstacle_accelerations))
                obstacle_positions, obstacle_velocities, enable_spawn, count_move = generate_obs_state_h_fixed(
                    obstacle_positions, obstacle_velocities, cycles, enable_spawn, ctrl.model, ctrl.data,
                    tool_frame_id, end_eff_pos, Dtrajectory_time, count_move, d_objective, v_ref, spawn_freq,
                    ee_vel)  # nominal_q, nominal_Dq, nominal_DDq)

            else:
                obstacle_positions, obstacle_velocities, obstacle_accelerations = bridge.getObstacles(elapsed=t)
            # print ("obstacle_positions:", obstacle_positions)
            # print ("type(obstacle_positions):", type(obstacle_positions))
            # print("size(obstacle_positions): ", obstacle_positions.shape)
            cycles += 1

            out = ctrl.step(
                obs_pos=obstacle_positions,
                obs_vel=obstacle_velocities,
                obs_acc=obstacle_accelerations,
                nominal_q=nominal_q,
                nominal_Dq=nominal_Dq,
                nominal_DDq=nominal_DDq
            )

            if PLOT_LAMBDAS and t > 0.1:
                gamma_list.append(ctrl.cfg.gamma)
                lambda_pos_list.append(ctrl.cfg.lambda_pos)
                lambda_vel_list.append(ctrl.cfg.lambda_vel)
                lambda_acc_list.append(ctrl.cfg.lambda_acc)
                lambda_scaling_list.append(ctrl.cfg.lambda_scaling)
                t_list.append(t)

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
                    # print("Trajectory time: ", trajectory_time)
                    # print(f"T_total: {T_total}")
                    # print(f"actual scaling: {Dtrajectory_time}")
                    enable_lap_count = False
            else:
                enable_lap_count = True
                # print(f"actual lap: {int(trajectory_time % T_total)}")
            visualizer.update_vectors(out["h_min"], out["d_min"], out["vr_min"]-out["vh_min"], t,)
            elapsed = time.perf_counter() - loop_start
            ct_qp.append(elapsed)

            # --------------------------- INTEGRATION ----------------------------
            t += Tc
            end_eff_pos = out["end_effector_pos"]
            ee_vel = out["end_effector_vel"]
            vr_min = out["vr_min"]

            # ----------------------------- TIMING -------------------------------
            dist = []
            for i in range(len(cartesian_configs.values())):
                q_wp = list(cartesian_configs.values())[i]
                dist.append(np.linalg.norm(q_wp - end_eff_pos))
                if np.linalg.norm(q_wp - end_eff_pos) < 2e-03 and prec_target != i:
                    on_target_count += 1
                    prec_target = i
                    # print ("TARGET REACHED")
                    break
            # print("Min dist: ", np.min(dist))
            if np.min(dist) > 0.0:
                min_dist.append(np.min(dist))
            elapsed = time.perf_counter() - loop_start
            if cycles > 1:
                ct.append(elapsed)
                scaling_log.append(Dtrajectory_time)
                h_log.append(out["h_min"])
                trj_error_log.append(out["trajectory_error"])

            if out["h_min"] < 0 and out["vr_min"] < -1e-3:
                violations += 1
            sum_scale += out["Dtrajectory_time"]
            trajectory_error_sum += out["trajectory_error"]
            if out["Dtrajectory_time"] < scaling_threshold:
                low_scale_count += 1
            if cycles % 5000 == 0:
                print(f"STILL ALIVE! T: {t:.2f}s")
            rest = Tc - elapsed
            if rest > 0:

                time.sleep(0.0001)
            else:
                timeout_cycles += 1
            if unfeasible_string != "FEASIBLE":
                unfeasible_cnt += 1

    except KeyboardInterrupt:
        # request a graceful stop; loop condition will exit on next iteration
        stop_event.set()
    #

    # Call with your
    computation_times = np.array(ct)
    scaling_log = np.array(scaling_log)
    h_log = np.array(h_log)
    trj_error_log = np.array(trj_error_log)
    print(f"LAP COUNT: {lap_count}")

    on_target_rate = on_target_count / (n_wp * ((lap_count) + ((trajectory_time % T_total) / T_total)))
    lap_count = lap_count + ((trajectory_time % T_total) / T_total)
    print(f"average scaling = {np.mean(scaling_log)}")

    on_target_rate = on_target_count / (n_wp * ((lap_count) + ((trajectory_time % T_total) / T_total)))
    lap_count = lap_count + ((trajectory_time % T_total) / T_total)
    viol_rate = violations / max(1, cycles)
    mean_scale = sum_scale / max(1, cycles)
    mean_trajectory_error = trajectory_error_sum / max(1, cycles)
    low_scale_rate = low_scale_count / max(1, cycles)

    print(
        f"timeout cycles = {timeout_cycles} over {cycles}, percentage = {100.0 * timeout_cycles / cycles}, average = {np.mean(computation_times)}")
    print(f"unfeasible cycles = {unfeasible_cnt} over {cycles}, percentage = {100.0 * unfeasible_cnt / cycles}")
    print(f"LAP COUNT: {lap_count}")
    print("on target count: ", on_target_count)
    print(((trajectory_time % T_total) / T_total))
    print(f"WAYPOINTS REACHING PERCENTAGE: {on_target_rate * 100.0} %")
    print(f"VIOLATION RATE: {viol_rate}")
    print(f"MEAN SCALING: {mean_scale}")
    print(f"MEAN TRAJECTORY ERROR: {mean_trajectory_error}")

    visualizer.compute_mean_cov(True)

    # SAVING RESULTS
    if SAVE_DATA:
        file_path = '../resullts/simulation_data_dynamic_params_comparison.csv'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Intestazioni delle colonne (headers)
        headers = [
            "test_type",
            "h_cfg",
            "h_mean_test",
            'on_target_rate',
            'lap_count',
            'viol_rate',
            'mean_scale',
            'mean_trajectory_error',
            'low_scale_rate'
        ]

        # I dati da salvare (calcolati come nel tuo esempio)
        row_data = {
            "test_type": test_name,
            "h_cfg": test_code,
            "h_mean_test" : visualizer.h_mean,
            'on_target_rate': on_target_rate,
            'lap_count': lap_count,
            'viol_rate': viol_rate,
            'mean_scale': mean_scale,
            'mean_trajectory_error': mean_trajectory_error,
            'low_scale_rate': low_scale_rate

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

    if PLOT_LAMBDAS:
        plot_lambdas(t_list, gamma_list, lambda_pos_list, lambda_vel_list, lambda_acc_list, lambda_scaling_list)
        cfg.plot_lambdas()
        plt.show()
rclpy.init()

test_type = ["P"]
controller_type = [1, 0, 0, 0]
duration = 1000.0

# test_code = ["paper", "h_-0.1", "h_0.1", "h_0.25", "h_0.5",  "h_1.0"]
h_mean_ref = [0, -0.10, 0.10, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
for i in range(len(h_mean_ref)-1):
    test_type.append("O")
h_cfg = ["", "article", "1", "-0.1"]
v_cfg = ["", "article", "1", "1"]



h_std_dev = 0.1

v_ref = 1.0
spawn_freq = 10

test_name = ["dynamic_params", "Paper_params",  "params_1.0", "params_-0.1"] # TEST NAME IN THE RESULTS FILE
test_code = test_name.copy()
for i in range(0, len(controller_type)):
    for j in range(len(test_type)):
     run_experiment(test_type[j],duration, h_mean_ref[j],h_std_dev,v_ref,spawn_freq,test_name[i], test_code[i],
                   controller_type[i], h_cfg[i], v_cfg[i])








