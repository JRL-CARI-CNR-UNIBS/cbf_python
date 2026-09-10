"""Optuna optimization for Base Optimal CBF Controller parameters."""

from datetime import datetime
import time
from typing import Tuple

import numpy as np
import optuna
import pinocchio as pin
from sharework import loadSharework

from cbf_python.bridges.fake_bridge import FakeCommandBridge
from cbf_python.controllers.base_optimal_controller import BCFOptimalController, ControllerConfig
from cbf_python.trajectory.joint_interpolator import SegmentedJointTrap
from cbf_python.utils.config_loader import load_yaml, resolve_path
from cbf_python.utils.optimization_helpers import run_episode_with_timeout, save_data_multiobj
from cbf_python.utils.simulation_helpers import (
    HOME,
    UR10E_JOINTS,
    compute_cartesian_poses,
    compute_ee_pose,
    plan_path,
)


def run_episode(
    cfg: ControllerConfig,
    duration: float = 60.0,
    scaling_threshold: float = 0.5,
) -> Tuple[float, float, float, float]:
    model_wrapper = loadSharework(UR10E_JOINTS)
    model = model_wrapper.model
    data = model.createData()

    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True)

    bridge_config = load_yaml("bridges.yaml")
    fake_cfg = bridge_config.get("fake_bridge", {})
    csv_file = resolve_path(fake_cfg.get("csv_path", "skeleton_vectors/skeleton_vectors_23.csv"))
    cam_cfg = fake_cfg.get("camera_pose", {})
    quat = pin.Quaternion(*cam_cfg.get("quaternion", [0.83, 0.185, 0.513, 0.12]))
    quat.normalize()
    R = quat.toRotationMatrix()
    T_wc = pin.SE3(R, np.array(cam_cfg.get("type_0_translation", [-0.094, -0.93, 2.309])))

    bridge = FakeCommandBridge(UR10E_JOINTS, csv_path=str(csv_file), Tworld_to_cam=T_wc)

    q = HOME.copy()
    ctrl.reset_state(q)
    planner = SegmentedJointTrap(Dq_max=cfg.Dq_max * 0.25, DDq_max=cfg.DDq_max * 0.125)
    plan_path(planner, q)
    T_total = planner.computeTime()

    traj_cart_error_log = []
    t = 0.0
    trajectory_time = 0.0
    violations = 0
    nsteps = 0
    sum_scale = 0.0
    trajectory_error_sum = 0.0

    while t < duration:
        obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed=t)
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
            end_eff_pos = out["end_effector_pos"]
            end_eff_nominal = out["Tbt_nominal"].translation
            cart_err = float(np.linalg.norm(end_eff_pos - end_eff_nominal))
            traj_cart_error_log.append(cart_err)
        except Exception:
            return 1.0, 0.0, 10.0, 1.0

        t += cfg.Tc
        trajectory_time = out["trajectory_time"]
        nsteps += 1

        if out["h_min"] < 0 and out["vr_min"] < -1e-3:
            violations += 1
        sum_scale += out["Dtrajectory_time"]
        trajectory_error_sum += out["trajectory_error"]

    traj_cart_arr = np.array(traj_cart_error_log)
    tv_cart = float(np.sum(np.abs(np.diff(traj_cart_arr))) / max(1, nsteps)) if len(traj_cart_arr) > 1 else 0.0
    viol_rate = violations / max(1, nsteps)
    mean_scale = sum_scale / max(1, nsteps)
    mean_traj_err = trajectory_error_sum / max(1, nsteps)

    return viol_rate, mean_scale, mean_traj_err, tv_cart * 1000.0


def make_objective(opt_cfg: dict):
    ep_cfg = opt_cfg.get("episode", {})
    Tc = float(ep_cfg.get("Tc", 0.002))
    duration = float(ep_cfg.get("duration", 60.0))
    delta = float(ep_cfg.get("delta_deg", 4.5))
    timeout = float(opt_cfg.get("study", {}).get("timeout", 600.0))

    ranges = opt_cfg.get("ranges", {})

    def objective(trial: optuna.Trial):
        cfg = ControllerConfig(Tc=Tc)
        p_range = ranges.get("lambda_pos", [100.0, 100000.0])
        v_range = ranges.get("lambda_vel", [0.001, 1000.0])
        a_range = ranges.get("lambda_acc", [1e-15, 1e-4])
        s_range = ranges.get("lambda_scaling", [10.0, 1000.0])
        g_range = ranges.get("gamma", [0.1, 20.0])

        cfg.lambda_pos = trial.suggest_float("lambda_pos", p_range[0], p_range[1], log=True)
        cfg.lambda_vel = trial.suggest_float("lambda_vel", v_range[0], v_range[1], log=True)
        cfg.lambda_acc = trial.suggest_float("lambda_acc", a_range[0], a_range[1], log=True)
        cfg.lambda_scaling = trial.suggest_float("lambda_scaling", s_range[0], s_range[1], log=True)
        cfg.gamma = trial.suggest_float("gamma", g_range[0], g_range[1], log=True)

        cfg.delta_q_max[0:2] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta)
        cfg.delta_q_max[2:4] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 2
        cfg.delta_q_max[4:6] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 4

        try:
            viol_rate, mean_scale, mean_traj_err, tv_cart = run_episode_with_timeout(
                run_episode, cfg=cfg, duration=duration, timeout=timeout
            )
            trial.set_user_attr("violation_rate", viol_rate)
        except TimeoutError:
            return 1.0, 0.0, 1.0

        return tv_cart, mean_scale, mean_traj_err

    return objective


def main(config_file: str = "optimization.yaml") -> None:
    opt_cfg = load_yaml(config_file)
    db_cfg = opt_cfg.get("database", {})
    study_cfg = opt_cfg.get("study", {})

    storage = None
    try:
        storage = optuna.storages.RDBStorage(url=db_cfg.get("url", ""))
    except Exception:
        if db_cfg.get("use_sqlite_fallback", True):
            sqlite_path = db_cfg.get("sqlite_path", "optuna_study.db")
            storage = f"sqlite:///{sqlite_path}"
            print(f"Using SQLite fallback storage: {storage}")

    study = optuna.create_study(
        directions=["minimize", "maximize", "minimize"],
        storage=storage,
        study_name=f"{study_cfg.get('name_prefix', 'params_optimal')}_{time.strftime('%Y%m%d-%H%M%S')}",
        sampler=optuna.samplers.NSGAIIISampler(),
        load_if_exists=True,
    )
    study.set_metric_names(["mean_tv_cartesian", "mean_scaling", "mean_trajectory_error"])

    n_trials = int(study_cfg.get("n_trials", 100))
    n_jobs = int(study_cfg.get("n_jobs", 1))

    print(f"Starting Optuna Study '{study.study_name}' for {n_trials} trials...")
    study.optimize(make_objective(opt_cfg), n_trials=n_trials, n_jobs=n_jobs)

    weights_dict = opt_cfg.get("weights", {})
    weights = [
        float(weights_dict.get("tv_cartesian", 1.0)),
        float(weights_dict.get("scaling", 1.0)),
        float(weights_dict.get("trajectory_error", 1.0)),
    ]
    save_data_multiobj(study, filename="log_best_trials.csv", n_samples=5, weights=weights)


if __name__ == "__main__":
    main()
