"""Optuna optimization for Base Optimal CBF Controller parameters.

This script executes multi-objective optimization using NSGA-III over four balanced metrics:
    1. Safety Penetration (minimize): Cumulative depth of barrier boundary violation [m]
    2. Productivity (maximize): Mean progression rate along the nominal path (dot{xi})
    3. Smoothness & Anti-Oscillation (minimize): Physical joint jerk norm and setpoint ringing
    4. Trajectory Tracking Error (minimize): Mean Euclidean error in joint configuration
"""

from datetime import datetime
import time
from typing import Optional, Tuple

import numpy as np
import optuna
import pinocchio as pin
from sharework import loadSharework

from cbf_python.bridges.fake_bridge import FakeCommandBridge, MultiRegionGestureSequencer
from cbf_python.controllers.base_optimal_controller import BCFOptimalController, ControllerConfig
from cbf_python.trajectory.joint_interpolator import SegmentedJointTrap
from cbf_python.utils.config_loader import load_yaml, populate_controller_config, resolve_path
from cbf_python.utils.metrics import compute_dynamic_risk_index
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
) -> Tuple[float, float, float, float, float, float, float, float, float, float, int, float, float]:
    """
    Execute a single simulation trial for multi-objective parameter optimization.

    Returns:
        Tuple containing:
            - mean_penetration: Mean ISO 15066 dynamic safety penetration depth [m] (minimize)
            - mean_scale: Average progression rate dot{xi} [0, 1] (maximize)
            - smoothness: Composite physical jerk and recovery oscillation penalty (minimize)
            - mean_traj_err: Mean joint tracking error norm [rad] (minimize)
            - viol_rate: Fraction of control steps penetrating ISO protective distance
            - worst_s_index: Worst-case SSM dynamic safety margin across episode [m]
            - rms_jerk: Root Mean Square joint jerk norm [rad/s^3]
            - mean_osc: Velocity tracking error sign reversal oscillation index
            - max_slack: Maximum soft-tube slack norm [rad]
            - mean_slack: Mean soft-tube slack norm [rad]
            - unfeasible_count: Count of control steps triggering emergency fallback
            - mean_cart_err: Mean Cartesian tracking error of the tool [m]
            - lap_count: Total fractional loops completed along nominal path
    """
    model_wrapper = loadSharework(UR10E_JOINTS)
    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True)

    bridge_config = load_yaml("bridges.yaml")
    fake_cfg = bridge_config.get("fake_bridge", {})
    csv_file = resolve_path(fake_cfg.get("csv_path", "skeleton_vectors/skeleton_vectors_23.csv"))
    cam_cfg = fake_cfg.get("camera_pose", {})
    quat = pin.Quaternion(*cam_cfg.get("quaternion", [0.83, 0.185, 0.513, 0.12]))
    quat.normalize()
    R = quat.toRotationMatrix()
    T_wc = pin.SE3(R, np.array(cam_cfg.get("type_0_translation", [-0.094, -0.93, 2.309])))

    bridge = FakeCommandBridge(
        UR10E_JOINTS,
        csv_path=str(csv_file),
        Tworld_to_cam=T_wc,
        slowdown_factor=float(fake_cfg.get("slowdown_factor", 1.0)),
    )

    q = HOME.copy()
    ctrl.reset_state(q)
    planner = SegmentedJointTrap(Dq_max=cfg.Dq_max * 0.25, DDq_max=cfg.DDq_max * 0.125)
    plan_path(planner, q)
    T_total = planner.computeTime()

    t = 0.0
    trajectory_time = 0.0
    violations = 0
    unfeasible_count = 0
    nsteps = 0
    sum_scale = 0.0
    trajectory_error_sum = 0.0
    sum_cart_error = 0.0
    sum_penetration = 0.0
    sum_jerk_sq = 0.0
    sum_osc = 0.0
    max_slack = 0.0
    sum_slack = 0.0
    min_s_index = float("inf")

    u_prev = np.zeros(ctrl.nq, dtype=np.float64)
    dot_e_prev = np.zeros(ctrl.nq, dtype=np.float64)
    margin_C = cfg.C_0 if cfg.C_0 is not None else cfg.C

    # Obstacle Playback Mode Configuration
    mode = str(fake_cfg.get("mode", "multi_region")).lower()
    sequencer: Optional[MultiRegionGestureSequencer] = None
    if mode == "multi_region":
        sequencer = MultiRegionGestureSequencer(bridge, gestures=fake_cfg.get("gestures"), T_total=T_total)

    target_xi = float(fake_cfg.get("target_xi", 0.475))
    lead_time = float(fake_cfg.get("lead_time", 2.80))
    gesture_duration = float(fake_cfg.get("gesture_duration", 7.0))
    rearm_for_next_lap = bool(fake_cfg.get("rearm_for_next_lap", True))

    target_s = target_xi * T_total
    triggered = False
    t_trigger = 0.0
    curr_scaling = 1.0

    while t < duration:
        if sequencer is not None:
            obs_pos, obs_vel, obs_acc, _ = sequencer.step(t, trajectory_time, curr_scaling)
        elif mode == "predictive_trigger":
            curr_s = trajectory_time % T_total
            v_est = max(0.05, curr_scaling)
            if not triggered:
                if curr_s < target_s:
                    t_eta = (target_s - curr_s) / v_est
                else:
                    t_eta = (target_s + T_total - curr_s) / v_est

                if t_eta <= lead_time:
                    triggered = True
                    t_trigger = t

            if triggered:
                tau = t - t_trigger
                if gesture_duration > 0.0 and tau >= gesture_duration:
                    if rearm_for_next_lap and (curr_s < target_s - 1.0 or curr_s > target_s + 2.0):
                        triggered = False
                        obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed=0.0)
                        obs_vel = np.zeros_like(obs_vel)
                        obs_acc = np.zeros_like(obs_acc)
                    else:
                        obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed=0.0)
                        obs_vel = np.zeros_like(obs_vel)
                        obs_acc = np.zeros_like(obs_acc)
                else:
                    obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed=tau)
            else:
                obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed=0.0)
                obs_vel = np.zeros_like(obs_vel)
                obs_acc = np.zeros_like(obs_acc)
        else:
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
            curr_scaling = float(out["Dtrajectory_time"])
            # 1. Kinematics and Cartesian tracking
            end_eff_pos = out["end_effector_pos"]
            end_eff_vel = out["end_effector_vel"]
            end_eff_nominal = out["Tbt_nominal"].translation
            cart_err = float(np.linalg.norm(end_eff_pos - end_eff_nominal))
            sum_cart_error += cart_err

            # 2. Dynamic SSM protective distance margin (ISO 15066 benchmark)
            s_index = compute_dynamic_risk_index(
                end_eff_pos=end_eff_pos,
                end_eff_vel=end_eff_vel,
                obs_positions=obs_pos,
                obs_velocities=obs_vel,
                obs_accelerations=obs_acc,
                a_s=cfg.a_s,
                T_r=cfg.Tr,
                delta=1.25,
                D_0=margin_C,
            )
            if s_index < min_s_index:
                min_s_index = s_index

            # Safety penetration below the ISO 15066 protective separation
            if s_index < 0.0:
                sum_penetration += (-s_index)
                violations += 1

            # 3. Physical Joint Jerk (du/dt)
            ddq = out["ddq"]
            dot_e = out["dq"] - nominal_Dq
            if nsteps > 0:
                jerk = (ddq - u_prev) / cfg.Tc
                sum_jerk_sq += float(np.sum(jerk ** 2))

                # 4. Velocity error sign reversals (quantifies setpoint ringing/oscillation)
                # Reversal occurs when dot_e[j] * dot_e_prev[j] < 0
                osc = float(np.sum(np.maximum(0.0, -dot_e * dot_e_prev)))
                sum_osc += osc

            dot_e_prev = dot_e.copy()
            u_prev = ddq.copy()

            # 5. Solver Feasibility & Soft-Tube Slack
            if out["unfeasible_cnt"] != "FEASIBLE":
                unfeasible_count += 1

            if "slack" in out and out["slack"] is not None:
                slack_mag = float(np.max(np.abs(out["slack"])))
                if slack_mag > max_slack:
                    max_slack = slack_mag
                sum_slack += slack_mag

        except Exception as err:
            # Fallback on numerical failure or integration divergence
            return 10.0, 0.0, 1e4, 10.0, 1.0, -10.0, 1e4, 10.0, 10.0, 10.0, 1000, 10.0, 0.0

        t += cfg.Tc
        trajectory_time = out["trajectory_time"]
        nsteps += 1
        sum_scale += float(out["Dtrajectory_time"])
        trajectory_error_sum += float(out["trajectory_error"])

    steps_safe = max(1, nsteps)
    mean_penetration = sum_penetration / steps_safe
    mean_scale = sum_scale / steps_safe
    mean_traj_err = trajectory_error_sum / steps_safe
    mean_cart_err = sum_cart_error / steps_safe

    # Physical RMS Jerk and Oscillation Index
    rms_jerk = float(np.sqrt(sum_jerk_sq / max(1, nsteps - 1))) if nsteps > 1 else 0.0
    mean_osc = float(sum_osc / max(1, nsteps - 1)) if nsteps > 1 else 0.0
    smoothness = rms_jerk + 100.0 * mean_osc

    viol_rate = violations / steps_safe
    worst_s_index = min_s_index if min_s_index != float("inf") else 10.0
    worst_s_index = float(np.clip(worst_s_index, -10.0, 10.0))
    mean_slack = sum_slack / steps_safe
    lap_count = float(trajectory_time / max(1e-6, T_total))

    return (
        mean_penetration,
        mean_scale,
        smoothness,
        mean_traj_err,
        viol_rate,
        worst_s_index,
        rms_jerk,
        mean_osc,
        max_slack,
        mean_slack,
        unfeasible_count,
        mean_cart_err,
        lap_count,
    )


def make_objective(opt_cfg: dict):
    ep_cfg = opt_cfg.get("episode", {})
    Tc = float(ep_cfg.get("Tc", 0.002))
    duration = float(ep_cfg.get("duration", 60.0))
    delta = float(ep_cfg.get("delta_deg", 4.5))
    timeout = float(opt_cfg.get("study", {}).get("timeout", 600.0))

    ranges = opt_cfg.get("ranges", {})
    ctrl_overrides = opt_cfg.get("controller", {})

    try:
        base_defaults = load_yaml("controller_defaults.yaml")
    except Exception:
        base_defaults = {}

    def objective(trial: optuna.Trial):
        cfg = ControllerConfig()
        populate_controller_config(cfg, {**base_defaults, **ctrl_overrides})
        cfg.Tc = Tc

        p_range = [float(x) for x in ranges.get("lambda_pos", [100.0, 100000.0])]
        v_range = [float(x) for x in ranges.get("lambda_vel", [0.01, 100.0])]
        a_range = [float(x) for x in ranges.get("lambda_acc", [1e-11, 1e-4])]
        s_range = [float(x) for x in ranges.get("lambda_scaling", [5.0, 500.0])]
        g_range = [float(x) for x in ranges.get("gamma", [0.5, 15.0])]

        cfg.lambda_pos = trial.suggest_float("lambda_pos", p_range[0], p_range[1], log=True)
        cfg.lambda_vel = trial.suggest_float("lambda_vel", v_range[0], v_range[1], log=True)
        cfg.lambda_acc = trial.suggest_float("lambda_acc", a_range[0], a_range[1], log=True)
        cfg.lambda_scaling = trial.suggest_float("lambda_scaling", s_range[0], s_range[1], log=True)
        cfg.gamma = trial.suggest_float("gamma", g_range[0], g_range[1], log=True)

        if cfg.use_soft_tube or "lambda_tube" in ranges:
            t_range = [float(x) for x in ranges.get("lambda_tube", [1.0e3, 1.0e7])]
            cfg.lambda_tube = trial.suggest_float("lambda_tube", t_range[0], t_range[1], log=True)

        cfg.delta_q_max[0:2] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta)
        cfg.delta_q_max[2:4] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 2
        cfg.delta_q_max[4:6] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 4

        try:
            (
                mean_penetration,
                mean_scale,
                smoothness,
                mean_traj_err,
                viol_rate,
                worst_s_index,
                rms_jerk,
                mean_osc,
                max_slack,
                mean_slack,
                unfeasible_count,
                mean_cart_err,
                lap_count,
            ) = run_episode_with_timeout(
                run_episode, cfg=cfg, duration=duration, timeout=timeout
            )
            trial.set_user_attr("violation_rate", viol_rate)
            trial.set_user_attr("min_s_index", worst_s_index)
            trial.set_user_attr("rms_jerk", rms_jerk)
            trial.set_user_attr("mean_osc", mean_osc)
            trial.set_user_attr("max_slack", max_slack)
            trial.set_user_attr("mean_slack", mean_slack)
            trial.set_user_attr("unfeasible_count", unfeasible_count)
            trial.set_user_attr("cartesian_error_mean", mean_cart_err)
            trial.set_user_attr("lap_count", lap_count)
        except TimeoutError:
            return 10.0, 0.0, 1e4, 10.0
        except Exception:
            # Prevent unexpected worker termination or process errors from aborting the entire study
            return 10.0, 0.0, 1e4, 10.0

        return mean_penetration, mean_scale, smoothness, mean_traj_err

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
        directions=["minimize", "maximize", "minimize", "minimize"],
        storage=storage,
        study_name=f"{study_cfg.get('name_prefix', 'params_optimal')}_{time.strftime('%Y%m%d-%H%M%S')}",
        sampler=optuna.samplers.NSGAIIISampler(),
        load_if_exists=True,
    )
    study.set_metric_names([
        "safety_penetration",
        "mean_scaling",
        "smoothness_jerk",
        "mean_trajectory_error",
    ])

    n_trials = int(study_cfg.get("n_trials", 100))
    n_jobs = int(study_cfg.get("n_jobs", 1))

    print(f"Starting Optuna Study '{study.study_name}' for {n_trials} trials...")
    study.optimize(make_objective(opt_cfg), n_trials=n_trials, n_jobs=n_jobs)

    weights_dict = opt_cfg.get("weights", {})
    weights = [
        float(weights_dict.get("safety_penetration", 3.0)),
        float(weights_dict.get("scaling", 1.5)),
        float(weights_dict.get("smoothness", 1.0)),
        float(weights_dict.get("trajectory_error", 1.0)),
    ]
    save_data_multiobj(study, filename="log_best_trials.csv", n_samples=5, weights=weights)


if __name__ == "__main__":
    main()
