import optuna
import optunahub
import numpy as np
import time
import pinocchio as pin
from optuna import visualization

from scripts.util.joint_interpolator import SegmentedJointTrap
from scripts.util.gaussian_process_util import generate_obs_state_h_fixed, compute_required_d, generate_target_h, save_data_multiobj

from sharework import loadSharework
from Command_bridge.fake_command_bridge import FakeCommandBridge
from Controller.optimal_cbf_task_controller import BCFOptimalController, ControllerConfig
from multiprocessing import Process, Queue
from queue import Empty
from Controller.dynamic_params_controllers import (PolynomialControllerConfig, PolynomialOptimalController,
                                                   StocasticalControllerConfig, StocasticalOptimalController)
from pathlib import Path

from optuna.samplers import CmaEsSampler
from optuna.samplers import NSGAIIISampler

# Database connection (for dashboard)
POSTGRES_URL = "postgresql+psycopg2://optuna:optuna_pw@localhost:5432/optuna_db"

import pandas as pd
import os
from datetime import datetime
from scripts.util.mean_visualizer import StochasticCBFVisualizer


# v_ref = 0.8

spawn_freq = 10
trial_duration = 250.0
n_trials = 2000
std_dev= 0.15
# Esempio di utilizzo:
# save_data_custom_names(study)

# --- ESEMPIO DI UTILIZZO ---
# save_data_complete(study)

def make_objective( h_mean_ref=0.1, ref_std_dev=0.1, v_ref= 0.8):
    def objective(trial,):

        cfg = ControllerConfig(Tc=2e-3)

        cfg.lambda_pos = trial.suggest_float("lambda_pos", 100, 1e5, log=True)
        cfg.lambda_vel = trial.suggest_float("lambda_vel", 1e-3, 1e3, log=True)
        cfg.lambda_acc = trial.suggest_float("lambda_acc", 1e-15, 1e-4, log=True)
        cfg.lambda_scaling = trial.suggest_float("lambda_scaling", 10, 1e3, log=True)
        cfg.gamma = trial.suggest_float("gamma", 2, 10, log=True)

        delta = 4.5

        cfg.delta_q_max[0:2] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta)
        cfg.delta_q_max[2:4] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 2
        cfg.delta_q_max[4:6] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 4

        try:
            viol_rate, mean_scale, mean_traj_err, low_scale_rate, lap_count, h_mean, d_mean, v_mean, cov_matrix = run_episode_with_timeout(
                cfg=cfg, Tc=2e-3, duration=trial_duration,
                timeout=6000, h_mean_ref=h_mean_ref, ref_std_dev=ref_std_dev,v_ref=v_ref
            )
            trial.set_user_attr("covariance_matrix", cov_matrix.tolist())
            trial.set_user_attr("h_mean", h_mean)
            trial.set_user_attr("d_mean", d_mean)
            trial.set_user_attr("v_rel_mean", v_mean)

        except TimeoutError:
            # For directions: [minimize, maximize, minimize, minimize]
            return 1.0, 0.0, 1.0, 1.0, 0.0

        return  viol_rate, mean_scale, mean_traj_err, low_scale_rate, lap_count

    return objective


# ----------------- STATIC INITIALIZATION (only once) -----------------
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
    p = T_ee.translation  # 3D position
    R = T_ee.rotation  # 3x3 rotation matrix
    return p, R, T_ee


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


# -------------------- EVALUATION FUNCTION --------------------
def run_episode(Tc=2e-3, duration=500.0, cfg=ControllerConfig(), h_mean_ref=0.1, ref_std_dev=0.1, v_ref=0.8):
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
    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True, keypoint_to_log=-1)

    bridge = FakeCommandBridge(
        UR10E_JOINTS,
        csv_path="/home/nyquist/projects/cells_ws/src/zed_skeleton_kinematics/csv_files/skeleton_vectors_14_NORMAL_TEST1.csv",
        Tworld_to_cam=T_wc,
        slowdown_factor=1.0,
        t0=0.0
    )

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

    visualizer = StochasticCBFVisualizer()
    while t < duration:
        # obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed=t)
        nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time % T_total)

            # obstacle_positions, obstacle_velocities, enable_spawn, count_move = generate_obs_state(obstacle_positions, obstacle_velocities, cycles, enable_spawn, planner, trajectory_time, T_total, model, data, tool_frame_id, ee_pos, Dtrajectory_time, count_move)
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
                if np.linalg.norm(q_wp - end_eff_pos) < 2e-03 and prec_target != i:
                    on_target_count += 1
                    prec_target = i
                    break
        except Exception:
            # Penalize infeasible or divergent QP
            print("QP failed")
            return 1.0, -1.0, 10.0, 1.0
        t += Tc
        nsteps += 1
        if out["h_min"] <  (h_objective+1.5*ref_std_dev) and out["h_min"] >  (h_objective-1.5*ref_std_dev):

            if out["h_min"] < 0 and out["vr_min"] < -1e-3:
                violations += 1
            sum_scale += out["Dtrajectory_time"]
            trajectory_error_sum += out["trajectory_error"]
            if out["Dtrajectory_time"] < scaling_threshold:
                low_scale_count += 1
            visualizer.update_vectors(out["h_min"], out["d_min"], out["vr_min"] - out["vh_min"], t)

        time.sleep(1e-4)  # To avoid locking issues in multiprocessing

    # on_target_rate = on_target_count/(n_wp * ((lap_count)+ ((trajectory_time % T_total)/T_total)))
    lap_count = lap_count + ((trajectory_time % T_total) / T_total)
    viol_rate = violations / max(1, len(visualizer.h_vec))
    mean_scale = sum_scale / max(1, len(visualizer.h_vec))
    mean_trajectory_error = trajectory_error_sum / max(1, len(visualizer.h_vec))
    low_scale_rate = low_scale_count / max(1, len(visualizer.h_vec))

    visualizer.compute_mean_cov()

    # on_target_rate = on_target_count/(n_wp * ((lap_count)+ ((trajectory_time % T_total)/T_total)))
    return viol_rate, mean_scale, mean_trajectory_error, low_scale_rate, lap_count, visualizer.h_mean, visualizer.d_mean, visualizer.v_mean, visualizer.cov_matrix


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
    heartbeat_interval=30,  # worker pings DB every 30s
    grace_period=120,  # declare failed if no ping for 120s
    # failed_trial_callback=RetryFailedTrialCallback(max_retry=1),
)

sampler = NSGAIIISampler(
    population_size=50,      # Aumenta la popolazione
    mutation_prob=0.2,       # Aumenta la mutazione (default è basso)
    crossover_prob=0.95,     # Alto crossover
)


ref_std_dev = 0.1
par_values = {
    0: [0.5, 1],
    1: [0.0, 1],
    2: [1,1],
    }
ref_h_mean_vec   = [-0.1, 1.5] #[x / 10.0 for x in range(-1, 11)]
for key in par_values:
    h_mean = par_values[key][0]
    v_ref = par_values[key][1]
    study = optuna.create_study(
    directions=["minimize", "maximize","minimize","minimize","maximize"],
    storage=storage,
    # load_if_exists=True,
    # sampler=sampler,
    sampler =optunahub.load_module("samplers/auto_sampler").AutoSampler(),
    study_name=f"params_GPR_test_h_mean_{h_mean}_v_mean_{v_ref}_{time.strftime('%Y%m%d-%H%M%S')}",
    # study_name=f"params_test_{time.strftime('%Y%m%d-%H%M%S')}",
    # study_name=f"dynamic_params_polynomial_20260216-094358",
    load_if_exists=True,

    )
    study.set_metric_names(["violation_rate", "mean_scaling", "mean_trajectory_error", "low_scale_rate", "lap count"])
    study.optimize(make_objective(h_mean,std_dev, v_ref), n_trials=n_trials, show_progress_bar=True, n_jobs=30, gc_after_trial=True)
    save_data_multiobj(study)
# print (run_episode())


''' 
TRIAL BELLI delta variabile

4999
4780
4789
4845
4706
4630
4558
761
'''

''' 
TRIAL BELLI delta fisso

3083
3289
3978
4937
'''