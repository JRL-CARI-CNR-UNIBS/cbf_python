import math

import optuna
import optunahub
import numpy as np
import time
import pinocchio as pin
from scripts.util.joint_interpolator import SegmentedJointTrap
from sharework import loadSharework
from Command_bridge.fake_command_bridge import FakeCommandBridge
from Controller.optimal_cbf_task_controller import BCFOptimalController, ControllerConfig
from multiprocessing import Process, Queue
from queue import Empty
from Controller.dynamic_params_controllers import (PolynomialControllerConfig, PolynomialOptimalController,
                                              )
from scripts.util.test_utils import compute_ee_pose
from scripts.util.gaussian_process_util import save_data_multiobj
from pathlib import Path
import pandas as pd
from scripts.util.gaussian_process_util import generate_obs_state_h_fixed, compute_required_d, generate_target_h, save_data_multiobj
spawn_freq = 10
trial_duration = 1500.0
n_trials = 6000
# std_dev= 0.1
# h_mean_ref = 0.5
# ref_std_dev = 0.5
# v_ref = 1.0
def compute_scenario_cost(viol_rate, mean_scale, traj_err, lap_count):
    # 1. Normalize based on expected bounds (adjust denominators to your reality)
    norm_viol = viol_rate / 0.1 # Assuming 20 is worst-case
    norm_scale = 1.0 - mean_scale  # 1 is best, 0 is worst
    norm_err = traj_err / 0.05  # Assuming 0.05 is worst-case error
    norm_lap = 1.0 - (lap_count / 200.0)  # Assuming 200 is best-case laps

    # 2. Define importance weights (Tune these based on what you care about most)
    w_viol = 5.0
    w_scale = 2.0
    w_err = 1.0
    w_lap = 0.2

    # 3. Compute final scalar cost
    total_cost = (w_viol * norm_viol) + \
                 (w_scale * norm_scale) + \
                 (w_err * norm_err) + \
                 (w_lap * norm_lap)

    return total_cost
# Database connection (for dashboard)
POSTGRES_URL = "postgresql+psycopg2://optuna:optuna_pw@localhost:5432/optuna_db"
params_filename = "parameters_set.csv"


def make_objective():
    def objective(trial):

        cfg = PolynomialControllerConfig(Tc = 2e-3)

        cfg.h_t = 1.0
        cfg.lambda_0_pos = trial.suggest_float("lambda_0_pos", 10, 1e6, log=True)
        cfg.lambda_0_vel = trial.suggest_float("lambda_0_vel", 1, 1e4, log=True)
        cfg.lambda_0_acc = trial.suggest_float("lambda_0_acc", 1e-15, 1e-6, log=True)
        cfg.lambda_0_scaling = trial.suggest_float("lambda_0_scaling", 10, 1e3, log=True)
        cfg.gamma_0 = trial.suggest_float("gamma_0", 0.1, 10, log=True)

        cfg.lambda_f_pos = trial.suggest_float("lambda_f_pos", 10, 1e6, log=True)
        cfg.lambda_f_vel = trial.suggest_float("lambda_f_vel", 1, 1e4, log=True)
        cfg.lambda_f_acc = trial.suggest_float("lambda_f_acc", 1e-15, 1e-6, log=True)
        cfg.lambda_f_scaling = trial.suggest_float("lambda_f_scaling", 10, 1e3, log=True)
        cfg.gamma_f = trial.suggest_float("gamma_f", 0.1, 10, log=True)

        cfg.m_pos = trial.suggest_float("m_pos", 1, 10, log=True)
        cfg.m_vel = trial.suggest_float("m_vel", 1, 10, log=True)
        cfg.m_acc = trial.suggest_float("m_acc", 1, 10, log=True)
        cfg.m_scaling = trial.suggest_float("m_scaling", 1, 10, log=True)
        cfg.m_gamma = trial.suggest_float("m_gamma", 1, 10, log=True)

        cfg.n_pos = trial.suggest_float("n_pos", 1e-4, 10, log=True)
        cfg.n_vel= trial.suggest_float("n_vel", 1e-4, 10, log=True)
        cfg.n_acc = trial.suggest_float("n_acc", 1e-4, 10, log=True)
        cfg.n_scaling = trial.suggest_float("n_scaling", 1e-4, 10, log=True)
        cfg.n_gamma = trial.suggest_float("n_gamma", 1e-4, 10, log=True)

        cfg.w_pos = trial.suggest_float("w_pos", 1e-9, 10, log = True)
        cfg.w_vel = trial.suggest_float("w_vel", 1e-9, 10, log = True)
        cfg.w_acc = trial.suggest_float("w_acc", 1e-9, 10, log = True)
        cfg.w_scaling = trial.suggest_float("w_scaling",1e-9, 10, log = True)
        cfg.w_gamma = trial.suggest_float("w_gamma", 1e-9, 10, log = True)

        cfg.lambda_pos = cfg.lambda_0_pos
        cfg.lambda_vel = cfg.lambda_0_vel
        cfg.lambda_scaling = cfg.lambda_0_scaling
        cfg.lambda_acc = cfg.lambda_0_acc
        cfg.gamma = cfg.gamma_0
        delta = 4.5

        cfg.delta_q_max[0:2] = np.deg2rad(np.array([1,1], dtype=np.float64) * delta)
        cfg.delta_q_max[2:4] = np.deg2rad(np.array([1,1], dtype=np.float64) * delta)*2
        cfg.delta_q_max[4:6] = np.deg2rad(np.array([1,1], dtype=np.float64) * delta)*4
        cfg.normalize_parameters()
        if not cfg.check_config_integrity():
            print("Invalid configuration, skipping trial")
            print(cfg)
            raise optuna.TrialPruned()
        try:
              sv_metrics, h_low_metrics, h_high_metrics  = run_episode_with_timeout(
                cfg = cfg, Tc=2e-3, duration=trial_duration,
                timeout=6000
              )
              # 3. Save the raw metrics to the trial
              # Standard Scenario
              trial.set_user_attr("sv_viol_rate", sv_metrics[0])
              trial.set_user_attr("sv_mean_scale", sv_metrics[1])
              trial.set_user_attr("sv_traj_err", sv_metrics[2])
              trial.set_user_attr("sv_lap_count", sv_metrics[3])

              # High-Low Scenario
              trial.set_user_attr("h_low_viol_rate", h_low_metrics[0])
              trial.set_user_attr("h_low_mean_scale", h_low_metrics[1])
              trial.set_user_attr("h_low_traj_err", h_low_metrics[2])
              trial.set_user_attr("h_low_lap_count", h_low_metrics[3])

              # High-High Scenario
              trial.set_user_attr("h_high_viol_rate", h_high_metrics[0])
              trial.set_user_attr("h_high_mean_scale", h_high_metrics[1])
              trial.set_user_attr("h_high_traj_err", h_high_metrics[2])
              trial.set_user_attr("h_high_lap_count", h_high_metrics[3])

              # 4. Compute composite costs using your function
              cost_sv = compute_scenario_cost(*sv_metrics)
              cost_h_low = compute_scenario_cost(*h_low_metrics)
              cost_h_high = compute_scenario_cost(*h_high_metrics)
        except TimeoutError:
            # For directions: [minimize, maximize, minimize, minimize]
            raise optuna.TrialPruned()

        return cost_sv, cost_h_low, cost_h_high
    return objective

# Camera and bridge
# Build camera pose from your INITI snippet
quat = pin.Quaternion(0.83, 0.185, 0.513, 0.12)
quat.normalize()
R = quat.toRotationMatrix()

T_wc = pin.SE3(R, np.array([0.094, -0.93, 2.309]))

home = np.array([90, -140, 140, -90, 90, 0]) * np.pi / 180.0
UR10E_JOINTS = [
    "ur10e_shoulder_pan_joint",
    "ur10e_shoulder_lift_joint",
    "ur10e_elbow_joint",
    "ur10e_wrist_1_joint",
    "ur10e_wrist_2_joint",
    "ur10e_wrist_3_joint",
]

Tc = 2e-3
gen_cfg = ControllerConfig(Tc=Tc)
# # Basic planner reused across trials
q = home.copy()
q10 = np.array([31.0, -78.0, 115.0, -127.0, 86.0, -32.0]) * np.pi / 180.0
q20 = np.array([31.0, -83.0, 98.0, -110.0, 86.0, -32.0]) * np.pi / 180.0
q22 = np.array([40.0, -126.0, 141.0, -100.0, 86.0, 45.0]) * np.pi / 180.0
q25 = np.array([130.0, -100.0, 125.0, -115.0, 94.0, -20.0]) * np.pi / 180.0
q30 = np.array([136.0, -60.0, 90.0, -122.0, 90.0, 45.0]) * np.pi / 180.0
q40 = np.array([134.0, -65.0, 70.0, -90.0, 90.0, 45.0]) * np.pi / 180.0
gen_cfg.Dq_max = gen_cfg.Dq_max * 0.25
gen_cfg.DDq_max = gen_cfg.DDq_max * 0.2

n_wp = 6
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
model_wrapper = loadSharework(UR10E_JOINTS)
model = model_wrapper.model
tool_frame_name = "ur10e_wrist_3_joint"
tool_frame_id = model.getFrameId(tool_frame_name)
data = model.createData()
for name in cartesian_configs:
    p, R, T_ee = compute_ee_pose(configs[name], model, data, tool_frame_id)
    cartesian_configs[name] = p.tolist()

scaling_threshold = 0.5

#-------------------- EVALUATION FUNCTION --------------------
def run_episode(Tc=2e-3, duration=500.0, cfg = PolynomialControllerConfig() ):

    home = np.array([90.0, -140.0, 140.0, -90.0, 90.0, 0.0]) * np.pi / 180.0

    cfg.Dq_max = cfg.Dq_max * 0.25
    cfg.DDq_max = cfg.DDq_max * 0.2
    planner = SegmentedJointTrap(Dq_max=gen_cfg.Dq_max * 0.25, DDq_max=gen_cfg.DDq_max * 0.25)
    # CONFIG 1
    planner.addWayPoint(q)
    planner.addWayPoint(q10)
    planner.addWayPoint(q22)
    planner.addWayPoint(q25)
    planner.addWayPoint(q30)
    planner.addWayPoint(q)


    T_total = planner.computeTime()
    model_wrapper = loadSharework(UR10E_JOINTS)
    model = model_wrapper.model
    data = model.createData()

    ctrl = PolynomialOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True, keypoint_to_log=-1)


    def run_subepisode(use_skeleton: bool = False, sub_duration = duration/3, h_mean_ref = 0.0, ref_std_dev=0.0, v_ref = 0.0):
        ctrl.reset_state(q)
        t = 0.0
        trajectory_time = 0.0
        violations, nsteps = 0, 0
        sum_scale = 0.0
        trajectory_error_sum = 0.0
        lap_count = 0
        on_target_count = 0
        prec_target = -1
        enable_lap_count = True

        low_scale_count = 0
        obstacle_positions = np.zeros(3)
        obstacle_velocities = np.zeros(3)
        obstacle_accelerations = np.array([20.0,20.0,20.0])*0.0
        obstacle_accelerations = obstacle_accelerations.reshape(1, 3)
        enable_spawn = True
        vr_min = -0.1
        ee_pos = np.zeros(3)
        ee_vel = np.zeros(3)
        count_move = 0
        Dtrajectory_time = 1.0
        bridge = FakeCommandBridge(
            UR10E_JOINTS,
            csv_path= "/home/nyquist/projects/cells_ws/src/zed_skeleton_kinematics/csv_files/skeleton_vectors_23.csv",
            Tworld_to_cam=T_wc,
            slowdown_factor=1.0,
            t0=0.0
        )
        while t < sub_duration:
            # obstacle_positions, obstacle_velocities, obstacle_accelerations = bridge.getObstacles(elapsed = t)
            nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time % T_total)
            if use_skeleton:
                obstacle_positions, obstacle_velocities, obstacle_accelerations = bridge.getObstacles(elapsed=t)
            else:
                h_objective = generate_target_h(h_mean_ref, ref_std_dev)
                d_objective = compute_required_d(h_objective, vr_min, v_ref, np.linalg.norm(obstacle_accelerations) )
                obstacle_positions, obstacle_velocities, enable_spawn, count_move = generate_obs_state_h_fixed(obstacle_positions, obstacle_velocities, nsteps, enable_spawn, ctrl.model, ctrl.data, tool_frame_id, ee_pos, Dtrajectory_time, count_move, d_objective, v_ref, spawn_freq, ee_vel)#nominal_q, nominal_Dq, nominal_DDq)

            try:
                out = ctrl.step(
                    obs_pos=obstacle_positions,
                    obs_vel=obstacle_velocities,
                    obs_acc=obstacle_accelerations,
                    nominal_q=nominal_q,
                    nominal_Dq=nominal_Dq,
                    nominal_DDq=nominal_DDq,
                )
                trajectory_time = out["trajectory_time"]

                vr_min = out["vr_min"]
                end_eff_pos = out["end_effector_pos"]
                ee_pos = out["end_effector_pos"]
                ee_vel = out["end_effector_vel"]
                Dtrajectory_time = out["Dtrajectory_time"]
                if (trajectory_time % T_total) < Tc:
                    if enable_lap_count:
                        lap_count += 1
                        prec_target = -1
                        enable_lap_count = False
                else:
                    enable_lap_count = True
                for i in range(len(cartesian_configs.values())):
                    q_wp = list(cartesian_configs.values())[i]
                    if  np.linalg.norm(q_wp - end_eff_pos) < 2e-03 and prec_target != i:
                        on_target_count += 1
                        prec_target = i
                        break
            except Exception as e:
                # Penalize infeasible or divergent QP
                print("QP failed")
                print (e)
                return 1.0, -1.0, 10.0, 1.0,0.0
            t += Tc

            nsteps += 1

            if out["h_min"] < 0 and out["vr_min"] < -1e-3:
                violations += 1
            sum_scale += out["Dtrajectory_time"]
            trajectory_error_sum += out["trajectory_error"]
            if out["Dtrajectory_time"] < scaling_threshold:
                low_scale_count += 1

            time.sleep(1e-4)  # To avoid locking issues in multiprocessing


        #on_target_rate = on_target_count/(n_wp * ((lap_count)+ ((trajectory_time % T_total)/T_total)))
        lap_count = lap_count + ((trajectory_time % T_total)/T_total)
        viol_rate = violations / max(1, nsteps)
        mean_scale = sum_scale / max(1, nsteps)
        mean_trajectory_error = trajectory_error_sum / max(1, nsteps)
        low_scale_rate = low_scale_count / max(1, nsteps)
        return (
                float(viol_rate),
                float(mean_scale),
                float(mean_trajectory_error),
                float(low_scale_rate),
                float(lap_count)
            )

    sv_params = run_subepisode(True, duration/3 )
    h_low_params = run_subepisode(False, duration/3, -0.1, 0.2, 1.0 )
    h_high_params = run_subepisode(False, duration/3, 0.9, 0.2, 1.0)
    return [sv_params, h_low_params, h_high_params]

def _run_episode_worker(args, kwargs, q):
    """Runs run_episode and returns either ('ok', result) or ('err', repr(exception))."""
    try:
        result = run_episode(*args, **kwargs)
        q.put(("ok", result))
    except Exception as e:
        q.put(("err", repr(e)))

def run_episode_with_timeout(*args, timeout=600, **kwargs):
    """
    Run run_episode(...), but stop it if it takes longer than `timeout` seconds.
    Returns the tuple from run_episode on success.
    Raises TimeoutError if exceeded, or RuntimeError if the worker failed.
    """
    q = Queue()
    p = Process(target=_run_episode_worker, args=(args, kwargs, q), daemon=True)
    p.start()
    p.join(timeout)

    if p.is_alive():
        # Hard timeout: terminate the process and clean up
        p.terminate()
        p.join()
        raise TimeoutError(f"run_episode exceeded {timeout}s and was terminated")

    try:
        status, payload = q.get_nowait()
    except Empty:
        raise RuntimeError("Worker exited without returning a result (crash or early exit).")

    if status == "ok":
        return payload
    else:
        raise RuntimeError(f"run_episode failed in worker: {payload}")

# -------------------- OPTUNA OPTIMIZATION --------------------



storage = optuna.storages.RDBStorage(
    url=POSTGRES_URL,
    engine_kwargs={
        "pool_pre_ping": True,
        "pool_size": 40,
        "max_overflow": 20,
    },
    heartbeat_interval=30,     # worker pings DB every 30s
    grace_period=120,          # declare failed if no ping for 120s
    # failed_trial_callback=RetryFailedTrialCallback(max_retry=1),
)

sampler = optuna.samplers.NSGAIISampler(
    population_size=100,  # 100 trials per generation (60 generations total)
    crossover_prob=0.9,   # High probability to mix good weight traits
    mutation_prob=0.15    # Slightly higher mutation to prevent local minima over 6000 trials
)
study = optuna.create_study(
    directions=["minimize", "minimize", "minimize"],
    sampler=sampler,
    storage=storage,
    #load_if_exists=True,
    #study_name=f"dynamic_params_polynomial_{time.strftime('%Y%m%d-%H%M%S')}",
    study_name=f"dynamic_params_polynomial_multicase_{time.strftime('%Y%m%d-%H%M%S')}",
    load_if_exists=True,

)
study.set_metric_names(["cost_sv", "cost_low", "cost_high"])
study.optimize(make_objective(), n_trials=n_trials, show_progress_bar=True, n_jobs=30, gc_after_trial=True)
save_data_multiobj(study, filename="Dynamic_parameters_general_case_results.csv")
    # print (run_episode(1e3,1e3,1e3,1e-3,5,1))
