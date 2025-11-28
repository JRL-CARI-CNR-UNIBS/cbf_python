import optuna
import optunahub
import numpy as np
import math
import time
import pinocchio as pin
from joint_interpolator import SegmentedJointTrap
from sharework import loadSharework
from fake_command_bridge import FakeCommandBridge
from optimal_cbf_task_controller import BCFOptimalController, ControllerConfig
from plotly.io import show

# Database connection (for dashboard)
POSTGRES_URL = "postgresql+psycopg2://optuna:optuna_pw@localhost:5432/optuna_db"
# ----------------- STATIC INITIALIZATION (only once) -----------------


# Camera and bridge
R = pin.utils.rotate('z', 1.9) @ pin.utils.rotate('x', 1.57)
# Build camera pose from your INITI snippet
quat = pin.Quaternion(0.814, 0.178, 0.535, 0.137)
quat.normalize()
R = quat.toRotationMatrix()
# 
T_wc = pin.SE3(R, np.array([0.108, -0.883, 2.351]))

home = np.array([90, -140, 140, -90, 90, 0]) * np.pi / 180.0
Tc =2e-3
gen_cfg = ControllerConfig(Tc=Tc)
# # Basic planner reused across trials
planner = SegmentedJointTrap(Dq_max=gen_cfg.Dq_max*.3, DDq_max=gen_cfg.DDq_max*.3)
q = home.copy()
q2 = home.copy()
q2[1] = -np.pi * 0.5
q2[2] = np.pi * 0.5
q3 = np.array([ 40.0, -80.0, 100.0, -120.0, 90.0, 0.0])*np.pi/180.0
q4 = np.array([ 122.0, -70.0, 100.0, -120.0, 90.0, 0.0])*np.pi/180.0

# CONFIG 1
# planner.addWayPoint(q)
# planner.addWayPoint(home)

# planner.addWayPoint(q2)
# planner.addWayPoint(home)

# CONFIG 2
planner.addWayPoint(q)
planner.addWayPoint(q3)
planner.addWayPoint(home)        
planner.addWayPoint(q4)
planner.addWayPoint(q)

# planner.addWayPoint(home)
# planner.addWayPoint(home + np.deg2rad([0, 20, -20, 0, 0, 0]))
# planner.addWayPoint(home)
T_total = planner.computeTime()

# -------------------- EVALUATION FUNCTION --------------------
def run_episode(lambda1, lambda2, lambda3, lambda4, gamma, delta, Tc=2e-3, duration=40.0):

    home = np.array([90.0, -140.0, 140.0, -90.0, 90.0, 0.0]) * np.pi / 180.0

    cfg = ControllerConfig(Tc=Tc)
    cfg.lambda1 = lambda1
    cfg.lambda2 = lambda2
    cfg.lambda3 = lambda3
    cfg.lambda4 = lambda4
    cfg.delta_q_max = np.deg2rad(np.array([1,1,1,1,1,1], dtype=np.float64) * delta)
    cfg.gamma = gamma
    UR10E_JOINTS = [
        "ur10e_shoulder_pan_joint",
        "ur10e_shoulder_lift_joint",
        "ur10e_elbow_joint",
        "ur10e_wrist_1_joint",
        "ur10e_wrist_2_joint",
        "ur10e_wrist_3_joint",
    ]
    model_wrapper = loadSharework(UR10E_JOINTS)
    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg)
   
    # quat = pin.Quaternion(0.814, 0.178, 0.535, 0.137)
    # quat.normalize()
    # R = quat.toRotationMatrix()

    # T_wc = pin.SE3(R, np.array([0.108, -0.883, 2.351]))

    bridge = FakeCommandBridge(
        UR10E_JOINTS,
        csv_path="/home/nyquist/projects/cells_ws/src/zed_skeleton_kinematics/zed_skeleton_kinematics/skeleton_vectors.csv",
        Tworld_to_cam=T_wc,
        slowdown_factor=1.0,
    )
    first_joint_position = home

    # q = first_joint_position.copy()
    # q2 = home.copy()
    # q2[1] = -np.pi * 0.5
    # q2[2] = np.pi * 0.5

    # planner = SegmentedJointTrap(Dq_max=cfg.Dq_max*.3, DDq_max=cfg.DDq_max*.3)
    # planner.addWayPoint(q)
    # planner.addWayPoint(home)
    # planner.addWayPoint(q2)
    # planner.addWayPoint(home)
    # T_total = planner.computeTime()

    ctrl.reset_state(q)
    t = 0.0
    trajectory_time = 0.0
    violations, nsteps = 0, 0
    sum_scale = 0.0
    trajectory_error_sum = 0.0
    # planner = SegmentedJointTrap(Dq_max=cfg.Dq_max*.3, DDq_max=cfg.DDq_max*.3)
    # planner.addWayPoint(q)
    # planner.addWayPoint(home)
    # planner.addWayPoint(q2)
    # planner.addWayPoint(home)
    # T_total = planner.computeTime()

    while t < duration:
        obs_pos, obs_vel, obs_acc = bridge.getObstacles()
        nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time % T_total)
        try:
            out = ctrl.step(
                obs_pos=obs_pos,
                obs_vel=obs_vel,
                obs_acc=obs_acc,
                nominal_q=nominal_q,
                nominal_Dq=nominal_Dq,
                nominal_DDq=nominal_DDq,
            )
        except Exception:
            # Penalize infeasible or divergent QP
            print("QP failed")
            return 1.0, -1.0

        if out["h_min"] < 0 and np.linalg.norm(out["dq"])>1e-3:
            violations += 1
        # print(f"violations = {violations}, nsteps = {nsteps}, h_min = {out['h_min']}, Dtrajectory_time = {out['Dtrajectory_time']}")
        sum_scale += out["Dtrajectory_time"]
        nsteps += 1
        t += Tc
        trajectory_time = out["trajectory_time"]
        trajectory_error_sum += out["trajectory_error"]
        time.sleep(1e-3)



    viol_rate = violations / max(1, nsteps)
    mean_scale = sum_scale / max(1, nsteps)
    mean_trajectory_error = trajectory_error_sum / max(1, nsteps)
    return viol_rate, mean_scale, mean_trajectory_error

# -------------------- OPTUNA OPTIMIZATION --------------------
def objective(trial):
    l1 = trial.suggest_float("lambda1", 1, 1e5, log=True)
    l2 = trial.suggest_float("lambda2", 1e-3, 1e3, log=True)
    l3 = trial.suggest_float("lambda3", 1e-2, 1e3, log=True)
    l4 = trial.suggest_float("lambda4", 1e-14, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 2, 10, log=True)
    delta = trial.suggest_float("delta_deg", 1, 10, log=True)
    viol_rate, mean_scale, mean_trajectory_error = run_episode(l1, l2, l3, l4, gamma, delta)

    return viol_rate, mean_scale, mean_trajectory_error  # minimize violations, maximize scaling


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

study = optuna.create_study(
    directions=["minimize", "maximize","minimize"],
    sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
    storage=storage,
    load_if_exists=True,
    study_name=f"config_2_viol_rate_mean_scaling_traj_error_study_{time.strftime('%Y%m%d-%H%M%S')}",
)
study.optimize(objective, n_trials=2500, show_progress_bar=True, n_jobs=30)
# fig = optuna.visualization.plot_pareto_front(study, target_names=["Violation Rate", "Mean Time Scaling"])
# show(fig)

# print("Pareto-optimal trials:")
# for t in study.best_trials:
#     print(f"Trial {t.number} values={t.values} params={t.params}")

# # Save study
# import pandas as pd
# df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
# timestamp = time.strftime("%Y%m%d-%H%M%S")
# df.to_csv(f"results/optuna_config2_study_l123_pareto_{timestamp}.csv", index=False)