"""
Optuna Multi-Objective Grid Optimization for GPR Dataset Generation.

Executes 2-second in-flight encounter slices across an oversampled 2D grid
of (h_target, v_rel) interaction regimes using NSGA-III to populate a 4D Pareto front:
  - Maximize: Performance (Throughput & Path Tracking)
  - Minimize: Safety Compliance Penalty (Closing velocity towards human)
  - Minimize: Jerk / Smoothness (Cartesian TV error)
  - Minimize: Infeasibility Count (QP solver fallback occurrences)
"""

from __future__ import annotations
import argparse
from datetime import datetime
import itertools
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pinocchio as pin
from sharework import loadSharework

from cbf_python.controllers.base_optimal_controller import (
    BCFOptimalController,
    ControllerConfig,
)
from cbf_python.trajectory.joint_interpolator import SegmentedJointTrap
from cbf_python.utils.config_loader import load_yaml
from cbf_python.utils.metrics import compute_dynamic_risk_index
from cbf_python.utils.obstacle_generators import compute_required_d
from cbf_python.utils.optimization_helpers import run_episode_with_timeout
from cbf_python.utils.simulation_helpers import (
    HOME,
    UR10E_JOINTS,
    plan_path,
)


def get_cruising_state(
    planner: SegmentedJointTrap,
    t_cruise: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample cruising state from the nominal planner."""
    q_nom, dq_nom, ddq_nom = planner.getMotionLaw(t_cruise)
    return np.array(q_nom, dtype=np.float64), np.array(dq_nom, dtype=np.float64), np.array(ddq_nom, dtype=np.float64)


def run_slice_episode(
    cfg: ControllerConfig,
    h_target: float,
    v_target: float,
    t_cruise: float = 3.5,
    duration: float = 2.0,
    alpha_cart_error: float = 2.0,
    warning_h_threshold: float = 0.20,
) -> Tuple[float, float, float, int]:
    """Execute a 2.0s encounter slice starting at steady cruising velocity."""
    model_wrapper = loadSharework(UR10E_JOINTS)
    model = model_wrapper.model
    data = model.createData()

    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True, keypoint_to_log=-1)

    # Initialize nominal planner
    planner = SegmentedJointTrap(Dq_max=cfg.Dq_max * 0.25, DDq_max=cfg.DDq_max * 0.125)
    plan_path(planner, HOME.copy())
    T_total = planner.computeTime()

    # Fast-forward robot to cruising condition
    q_cruise, dq_cruise, _ = get_cruising_state(planner, t_cruise % T_total)
    ctrl.reset_state(q_cruise, dq_cruise)

    # Determine Cartesian state of the tool frame at t_start
    tool_frame_id = ctrl.tool_frame_id
    pin.forwardKinematics(model, data, q_cruise, dq_cruise)
    pin.updateFramePlacements(model, data)
    T_tool = data.oMf[tool_frame_id]
    p_ee = np.array(T_tool.translation, dtype=np.float64)

    twist = pin.getFrameVelocity(model, data, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    v_ee = np.array(twist.linear, dtype=np.float64)
    v_norm = float(np.linalg.norm(v_ee))

    # Unit vector along motion direction
    u_dir = v_ee / v_norm if v_norm > 1e-4 else np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Compute analytical required distance for target safety margin h
    # v_r along line of sight (opposing direction vector)
    v_r_proj = -v_norm
    margin_C = cfg.C_0 if getattr(cfg, "C_0", None) is not None else cfg.C
    d_req = compute_required_d(
        h=h_target,
        v_r=v_r_proj,
        v_h=v_target,
        a_h=0.0,
        Tr=cfg.Tr,
        a_s=cfg.a_s,
        C=margin_C,
    )
    d_req = max(0.01, d_req)

    # Spawn single obstacle in front of the robot's motion vector
    p_obs = p_ee + d_req * u_dir
    v_obs = -v_target * u_dir  # Closing velocity directed towards robot
    a_obs = np.zeros(3, dtype=np.float64)

    obs_positions = p_obs.reshape(1, 3)
    obs_velocities = v_obs.reshape(1, 3)
    obs_accelerations = a_obs.reshape(1, 3)

    # Slicing loop metrics
    t = 0.0
    trajectory_time = 0.0
    nsteps = 0
    sum_scale = 0.0
    sum_cart_error = 0.0
    min_s_index = float("inf")
    unfeasible_count = 0
    traj_cart_error_log: List[float] = []

    Tc = cfg.Tc

    while t < duration:
        current_traj_time = (t_cruise + trajectory_time) % T_total
        nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(current_traj_time)

        try:
            out = ctrl.step(
                obs_pos=obs_positions,
                obs_vel=obs_velocities,
                obs_acc=obs_accelerations,
                nominal_q=nominal_q,
                nominal_Dq=nominal_Dq,
                nominal_DDq=nominal_DDq,
            )
            ee_pos = out["end_effector_pos"]
            ee_vel = out["end_effector_vel"]
            ee_nom = out["Tbt_nominal"].translation
            cart_err = float(np.linalg.norm(ee_pos - ee_nom))
            traj_cart_error_log.append(cart_err)
            sum_cart_error += cart_err

            if out["unfeasible_cnt"] != "FEASIBLE":
                unfeasible_count += 1

            # Dynamic safety margin index (worst-case during encounter)
            s_index = compute_dynamic_risk_index(
                end_eff_pos=ee_pos,
                end_eff_vel=ee_vel,
                obs_positions=obs_positions,
                obs_velocities=obs_velocities,
                obs_accelerations=obs_accelerations,
                a_s=cfg.a_s,
                T_r=cfg.Tr,
                delta=1.25,
                D_0=cfg.C,
            )
            if s_index < min_s_index:
                min_s_index = s_index

        except Exception as err:
            # Solver failure/divergence fallback penalty
            # print(f"Error in slice: {err}")
            return -10.0, -10.0, 100.0, 1000

        # Propagate obstacle in simulation
        obs_positions[0] += obs_velocities[0] * Tc

        t += Tc
        trajectory_time = out["trajectory_time"]
        sum_scale += float(out["Dtrajectory_time"])
        nsteps += 1

    steps_safe = max(1, nsteps)
    mean_scale = sum_scale / steps_safe
    mean_cart_err = sum_cart_error / steps_safe

    # Bound safety index for clean numerical stability in Optuna
    worst_s_index = min_s_index if min_s_index != float("inf") else 10.0
    worst_s_index = float(np.clip(worst_s_index, -10.0, 10.0))

    # TV Cartesian error as jerk proxy
    err_arr = np.array(traj_cart_error_log)
    tv_cart = float(np.sum(np.abs(np.diff(err_arr))) / steps_safe) * 1000.0 if len(err_arr) > 1 else 0.0

    # Combined Task Performance Metric (maximize)
    perf_metric = mean_scale - alpha_cart_error * mean_cart_err

    return perf_metric, worst_s_index, tv_cart, unfeasible_count


def make_slice_objective(
    h_target: float,
    v_target: float,
    opt_cfg: Dict[str, Any],
):
    """Build Optuna multi-objective evaluation function for a single (h, v_rel) cell."""
    ep_cfg = opt_cfg.get("episode", {})
    Tc = float(ep_cfg.get("Tc", 0.002))
    duration = float(ep_cfg.get("duration", 2.0))
    alpha_cart = float(ep_cfg.get("alpha_cart_error", 2.0))
    warn_h = float(ep_cfg.get("warning_h_threshold", 0.20))
    t_cruise = float(ep_cfg.get("cruise_t_start", 3.5))
    timeout = float(opt_cfg.get("study", {}).get("timeout", 300.0))

    ranges = opt_cfg.get("ranges", {})

    def objective(trial: optuna.Trial) -> Tuple[float, float, float, int]:
        cfg = ControllerConfig(Tc=Tc)
        p_r = ranges.get("lambda_pos", [10.0, 100000.0])
        v_r = ranges.get("lambda_vel", [0.001, 1000.0])
        a_r = ranges.get("lambda_acc", [1e-15, 1e-4])
        s_r = ranges.get("lambda_scaling", [10.0, 1000.0])
        g_r = ranges.get("gamma", [0.1, 20.0])
        d_r = ranges.get("delta_deg", [1.0, 10.0])

        cfg.lambda_pos = trial.suggest_float("lambda_pos", p_r[0], p_r[1], log=True)
        cfg.lambda_vel = trial.suggest_float("lambda_vel", v_r[0], v_r[1], log=True)
        cfg.lambda_acc = trial.suggest_float("lambda_acc", a_r[0], a_r[1], log=True)
        cfg.lambda_scaling = trial.suggest_float("lambda_scaling", s_r[0], s_r[1], log=True)
        cfg.gamma = trial.suggest_float("gamma", g_r[0], g_r[1], log=True)
        delta_deg = trial.suggest_float("delta_deg", d_r[0], d_r[1], log=False)

        # Scale joint tube limits accordingly
        cfg.delta_q_max[0:2] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta_deg)
        cfg.delta_q_max[2:4] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta_deg) * 2.0
        cfg.delta_q_max[4:6] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta_deg) * 4.0

        try:
            perf, safety, tv_cart, unfeas = run_episode_with_timeout(
                run_slice_episode,
                cfg=cfg,
                h_target=h_target,
                v_target=v_target,
                t_cruise=t_cruise,
                duration=duration,
                alpha_cart_error=alpha_cart,
                warning_h_threshold=warn_h,
                timeout=timeout,
            )
        except TimeoutError:
            return -10.0, 50.0, 50.0, 1000

        # Store infeasibility count as user attribute as well
        trial.set_user_attr("unfeasible_count", int(unfeas))
        trial.set_user_attr("h_target", float(h_target))
        trial.set_user_attr("v_target", float(v_target))

        return perf, safety, tv_cart, int(unfeas)

    return objective


def main(config_file: str = "optimization_gpr.yaml") -> None:
    parser = argparse.ArgumentParser(description="Run Optuna GPR grid optimization")
    parser.add_argument("--config", type=str, default=config_file, help="Configuration YAML file")
    parser.add_argument("--max_points", type=int, default=None, help="Cap number of grid cells for testing")
    args = parser.parse_args()

    opt_cfg = load_yaml(args.config)
    db_cfg = opt_cfg.get("database", {})
    study_cfg = opt_cfg.get("study", {})
    grid_cfg = opt_cfg.get("grid", {})

    h_values = [float(x) for x in grid_cfg.get("h_values", [0.0, 0.2, 0.5])]
    v_values = [float(x) for x in grid_cfg.get("v_values", [0.0, 0.5, 1.0])]

    all_grid_points = list(itertools.product(h_values, v_values))
    if args.max_points is not None:
        all_grid_points = all_grid_points[: args.max_points]

    print(f"=== Starting GPR Grid Optimization across {len(all_grid_points)} Regimes ===")
    print(f"h values ({len(h_values)}): {h_values}")
    print(f"v values ({len(v_values)}): {v_values}")

    storage = None
    if db_cfg.get("use_sqlite_fallback", True):
        sqlite_path = db_cfg.get("sqlite_path", "optuna_gpr_grid.db")
        storage = f"sqlite:///{sqlite_path}"
        print(f"Using SQLite database storage: {storage}")
    else:
        storage = db_cfg.get("url", "")

    prefix = study_cfg.get("name_prefix", "gpr_slice")
    n_trials = int(study_cfg.get("n_trials", 35))
    n_jobs = int(study_cfg.get("n_jobs", 1))

    for idx, (h_t, v_t) in enumerate(all_grid_points):
        study_name = f"{prefix}_h_{h_t:+.3f}_v_{v_t:+.2f}".replace("+", "p").replace("-", "m")
        print(f"\n[{idx + 1}/{len(all_grid_points)}] Launching study '{study_name}' (h={h_t:.3f}, v={v_t:.2f})")

        study = optuna.create_study(
            directions=["maximize", "maximize", "minimize", "minimize"],
            storage=storage,
            study_name=study_name,
            sampler=optuna.samplers.NSGAIIISampler(),
            load_if_exists=True,
        )
        study.set_metric_names(["performance", "safety_index_min", "tv_cart", "unfeasible_count"])

        completed_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        needed_trials = max(0, n_trials - completed_count)

        if needed_trials > 0:
            study.optimize(
                make_slice_objective(h_target=h_t, v_target=v_t, opt_cfg=opt_cfg),
                n_trials=needed_trials,
                n_jobs=n_jobs,
            )
            print(f"Completed {needed_trials} new trials (total {len(study.trials)}) for '{study_name}'.")
        else:
            print(f"Study '{study_name}' already has {completed_count} completed trials. Skipping.")

    print("\n=== All Grid Optimization Studies Completed Successfully! ===")


if __name__ == "__main__":
    main()

