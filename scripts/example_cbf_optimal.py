#!/usr/bin/env python3
"""
Example Script: Bounded Control Barrier Function (B-CBF) Task Space Controller.

This script demonstrates the B-CBF optimal controller running on a 6-DOF UR10e robot
interacting with simulated or recorded human obstacles in real time:
- Generates a multi-waypoint periodic reference trajectory with trapezoidal velocity profiles.
- Solves a Quadratic Program (QP) at each cycle to enforce Speed and Separation Monitoring (SSM)
  Control Barrier Functions and bounding tube constraints.
- Visualizes the robot motion, obstacles, velocity vectors, and tracking metrics via Meshcat.
- Computes comprehensive safety, tracking, and efficiency statistics.
"""

import os
import csv
import time
import math
import yaml
import signal
import argparse
import threading
import functools
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pinocchio as pin
import meshcat.geometry as mgeom
from pinocchio.visualize import MeshcatVisualizer

from sharework import loadSharework
from Controller.optimal_cbf_task_controller import BCFOptimalController, ControllerConfig
from Command_bridge.joint_command_bridge import JointStateCommandBridge
from Command_bridge.fake_command_bridge import FakeCommandBridge

from scripts.util.joint_interpolator import SegmentedJointTrap
from scripts.util.visualization_daemon import VisualizationDaemon
from scripts.util.statistics_calculator import StatisticsCalculator
from scripts.util.bcf_utils import compute_dynamic_risk_index
from scripts.util.test_utils import (
    bring_robot_home,
    plan_path,
    compute_cartesian_poses,
)
from scripts.util import csv_publishers, test_publish_utils as pub_utils

stop_event = threading.Event()


def _handle_sigint(bridge, signum, frame):
    """Graceful SIGINT interrupt handler."""
    stop_event.set()
    if bridge is not None:
        try:
            bridge.shutdown()
        except Exception:
            pass


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads configuration parameters from a YAML file.
    Falls back to 'config/optimal_cbf_params.yaml' if no path is provided.
    """
    if config_path is None or not os.path.isfile(config_path):
        # Resolve path relative to project root or current working dir
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_yaml = os.path.join(base_dir, "config", "optimal_cbf_params.yaml")
        if os.path.isfile(default_yaml):
            config_path = default_yaml
        else:
            config_path = "config/optimal_cbf_params.yaml"

    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_controller(model_wrapper, config: Dict[str, Any]) -> Tuple[ControllerConfig, BCFOptimalController]:
    """Initializes ControllerConfig and BCFOptimalController using configuration dictionary."""
    sim_cfg = config.get("simulation", {})
    ctrl_cfg = config.get("controller", {})
    robot_cfg = config.get("robot", {})

    Tc = float(sim_cfg.get("control_period", 0.002))
    cfg = ControllerConfig(Tc=Tc)

    # Frame assignments
    cfg.prefix = str(robot_cfg.get("prefix", "ur10e_"))
    cfg.tool_frame = str(robot_cfg.get("tool_frame", "ur10e_wrist_3_joint"))
    cfg.elbow_frame = str(robot_cfg.get("elbow_frame", "ur10e_upper_arm_link"))
    cfg.shoulder_frame = str(robot_cfg.get("shoulder_frame", "ur10e_shoulder_link"))

    # CBF & safety parameters
    cfg.gamma = float(ctrl_cfg.get("gamma", 5.95))
    cfg.lambda_pos = float(ctrl_cfg.get("lambda_pos", 2098.0))
    cfg.lambda_vel = float(ctrl_cfg.get("lambda_vel", 0.343))
    cfg.lambda_scaling = float(ctrl_cfg.get("lambda_scaling", 16.56))
    cfg.lambda_acc = float(ctrl_cfg.get("lambda_acc", 1.455e-10))
    cfg.Tr = float(ctrl_cfg.get("Tr", 0.15))
    cfg.a_s = float(ctrl_cfg.get("a_s", 2.5))
    cfg.C = float(ctrl_cfg.get("C", 0.25))
    cfg.max_obstacles = int(ctrl_cfg.get("max_obstacles", 90))

    # Joint deviation tube bounds
    delta_deg = float(ctrl_cfg.get("delta_deg", 4.5))
    scales = ctrl_cfg.get("delta_q_scales", [1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
    for i in range(min(len(scales), 6)):
        cfg.delta_q_max[i] = np.deg2rad(delta_deg * float(scales[i]))

    use_cbf = bool(ctrl_cfg.get("use_cbf", True))
    keypoint_to_log = int(ctrl_cfg.get("keypoint_to_log", -1))

    ctrl = BCFOptimalController(
        model_wrapper=model_wrapper,
        cfg=cfg,
        useCbf=use_cbf,
        keypoint_to_log=keypoint_to_log,
    )
    return cfg, ctrl


def setup_visualization(model_wrapper, n_obstacles: int = 90):
    """Initializes Meshcat visualizer and background VisualizationDaemon."""
    viz = MeshcatVisualizer(
        model_wrapper.model,
        model_wrapper.collision_model,
        model_wrapper.visual_model,
    )
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Pre-render obstacle spheres (keypoint 7 highlighted in black)
    for i in range(n_obstacles):
        color = 0x000000 if i == 7 else 0xFF0000
        viz.viewer[f"obstacle_{i}"].set_object(
            mgeom.Sphere(0.1),
            mgeom.MeshLambertMaterial(color=color),
        )

    # Goal box
    side = 0.2
    viz.viewer["goal"].set_object(
        mgeom.Box([side, side, side / 10.0]),
        mgeom.MeshLambertMaterial(color=0x00FF00),
    )

    renderer = VisualizationDaemon(viz)
    return viz, renderer


def main(config_path: Optional[str] = None):
    # 0. Load Configuration
    config = load_config(config_path)

    sim_cfg = config.get("simulation", {})
    robot_cfg = config.get("robot", {})
    planner_cfg = config.get("planner", {})
    bridge_cfg = config.get("bridge", {})
    logging_cfg = config.get("logging", {})

    duration = float(sim_cfg.get("duration", 30.0))
    control_period = float(sim_cfg.get("control_period", 0.002))
    show_data = bool(sim_cfg.get("show_data", True))
    use_bridge = bool(sim_cfg.get("use_bridge", False))
    log_data = bool(sim_cfg.get("log_data", False))
    save_data = bool(sim_cfg.get("save_data", False))
    test_name = str(sim_cfg.get("test_name", "optimal_cbf_run"))

    joint_names = robot_cfg.get("joint_names", [
        "ur10e_shoulder_pan_joint",
        "ur10e_shoulder_lift_joint",
        "ur10e_elbow_joint",
        "ur10e_wrist_1_joint",
        "ur10e_wrist_2_joint",
        "ur10e_wrist_3_joint",
    ])
    home_config_deg = robot_cfg.get("home_config_deg", [90.0, -140.0, 140.0, -90.0, 90.0, 0.0])
    home_config = np.deg2rad(np.array(home_config_deg, dtype=float))

    print("=" * 70)
    print(f"Starting B-CBF Optimal Controller Execution: {test_name}")
    print(f"Mode: {'ROS 2 Live Bridge' if use_bridge else 'Offline Simulation Replay'}")
    print(f"Duration: {duration} s | Control Period: {control_period} s")
    print("=" * 70)

    # 1. Load Robot Kinematic & Dynamic Model
    model_wrapper = loadSharework(joint_names)
    model = model_wrapper.model

    # 2. Configure Controller
    cfg, ctrl = setup_controller(model_wrapper, config)
    print(cfg)

    # 3. Setup Command Bridge
    tool_frame_name = cfg.tool_frame
    if use_bridge:
        threshold = float(bridge_cfg.get("threshold", 1.1))
        timeout_sec = float(bridge_cfg.get("timeout_sec", 5.0))
        bridge = JointStateCommandBridge(
            ordered_joint_names=joint_names,
            threshold=threshold,
        )
        first_joint_position = bridge.wait_for_first_state(tool_frame_name, timeout=timeout_sec)
        signal.signal(signal.SIGINT, functools.partial(_handle_sigint, bridge))
        if math.isnan(first_joint_position):
            bridge.shutdown()
            return
        first_joint_position = bridge.getPositions()
        bridge.switch_to_forward_position_controller_service()
    else:
        offline_cfg = bridge_cfg.get("offline_dataset", {})
        csv_path = str(offline_cfg.get("csv_path", "../skeleton_vectors/skeleton_vectors_23.csv"))
        slowdown_factor = float(offline_cfg.get("slowdown_factor", 1.0))
        t0 = float(offline_cfg.get("t0", 0.0))

        cam_tf = offline_cfg.get("camera_transform", {})
        quat_wxyz = cam_tf.get("quaternion_wxyz", [0.83, 0.185, 0.513, 0.12])
        trans = cam_tf.get("translation", [-0.094, -0.93, 2.309])

        quat = pin.Quaternion(float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]))
        quat.normalize()
        T_wc = pin.SE3(quat.toRotationMatrix(), np.array(trans, dtype=float))

        bridge = FakeCommandBridge(
            joint_names,
            csv_path=csv_path,
            Tworld_to_cam=T_wc,
            slowdown_factor=slowdown_factor,
            t0=t0,
        )
        first_joint_position = home_config.copy()
        signal.signal(signal.SIGINT, functools.partial(_handle_sigint, None))

    # 4. Setup Visualization
    renderer = None
    if show_data:
        _, renderer = setup_visualization(model_wrapper, n_obstacles=cfg.max_obstacles)

    # 5. Bring Robot to Home Position & Plan Reference Trajectory
    q = first_joint_position.copy()
    if use_bridge:
        bring_robot_home(cfg, q, home_config, bridge, ctrl)
        q = home_config.copy()

    dq_scale = float(planner_cfg.get("dq_max_scale", 0.25))
    ddq_scale = float(planner_cfg.get("ddq_max_scale", 0.125))
    planner = SegmentedJointTrap(Dq_max=cfg.Dq_max * dq_scale, DDq_max=cfg.DDq_max * ddq_scale)
    plan_path(planner, q)
    T_total = planner.computeTime()
    print(f"Reference trajectory planned: {T_total:.2f} s per cycle")

    if show_data and renderer is not None:
        renderer.publishPath(planner.publishPath())

    cartesian_configs = compute_cartesian_poses(q, model)
    scaling_threshold = float(planner_cfg.get("scaling_threshold", 0.5))
    n_waypoints = int(planner_cfg.get("n_waypoints", 10))
    stats_calculator = StatisticsCalculator(
        n_wp=n_waypoints,
        T_total=T_total,
        cartesian_configs=cartesian_configs,
        Tc=control_period,
        scaling_threshold=scaling_threshold,
    )

    # 6. Setup Logging if requested
    log_publishers = {}
    if log_data:
        results_dir = str(logging_cfg.get("results_dir", "../results/simulation/scaling"))
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_path = os.path.join(results_dir, now_str)
        os.makedirs(test_path, exist_ok=True)
        log_publishers["target"] = csv_publishers.JointTargetCsvPublisher(
            csv_path=f"{test_path}/reference_trajectory.csv",
            column_names="time," + ",".join([f"target_joint_{i}_{attr}" for i in range(len(joint_names)) for attr in ["pos", "vel", "acceleration"]]),
            joint_names=joint_names,
        )
        log_publishers["state"] = csv_publishers.JointTargetCsvPublisher(
            csv_path=f"{test_path}/joint_states.csv",
            column_names="time," + ",".join([f"joint_{i}_{attr}" for i in range(len(joint_names)) for attr in ["pos", "vel", "acceleration"]]),
            joint_names=joint_names,
        )
        log_publishers["cbf"] = csv_publishers.DoubleArrayCsvPublisher(
            csv_path=f"{test_path}/cbf_results.csv",
            column_names="time,h_min,d_min,trajectory_error,pos_ee_x,pos_ee_y,pos_ee_z,vel_ee_x,vel_ee_y,vel_ee_z,v_r_min,v_h_min,scaling",
        )
        log_publishers["start"] = csv_publishers.TestStartCsvPublisher(
            csv_path=f"{test_path}/TEST_START.csv",
            column_names="time,val",
        )
        log_publishers["start"].publish_once(True)

    # 7. Main Real-Time Control Loop
    t = 0.0
    trajectory_time = 0.0
    ctrl.reset_state(q)

    try:
        while t < duration and not stop_event.is_set():
            loop_start = time.perf_counter()

            # A. Get Obstacles
            if use_bridge:
                obs_pos, obs_vel, obs_acc = bridge.getObstacles()
            else:
                obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed=t)

            # B. Get Reference Waypoint
            nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time % T_total)

            # C. Step B-CBF Optimal Controller
            out = ctrl.step(
                obs_pos=obs_pos,
                obs_vel=obs_vel,
                obs_acc=obs_acc,
                nominal_q=nominal_q,
                nominal_Dq=nominal_Dq,
                nominal_DDq=nominal_DDq,
            )

            q = out["q"]
            dq = out["dq"]
            ddq = out["ddq"]
            trajectory_time = out["trajectory_time"]
            unfeasible_status = out["unfeasible_cnt"]

            # D. Send Commands to Actuators
            if use_bridge and not stop_event.is_set():
                bridge.sendCommand(q)

            # E. Metrics & Diagnostics
            end_eff_pos = out["end_effector_pos"]
            end_eff_vel = out["end_effector_vel"]
            end_eff_nom_pos = out["Tbt_nominal"].translation
            trajectory_cart_err = float(np.linalg.norm(end_eff_pos - end_eff_nom_pos))

            s_index = None
            if stats_calculator.cycles > 1:
                s_index = compute_dynamic_risk_index(
                    end_eff_pos=end_eff_pos,
                    end_eff_vel=end_eff_vel,
                    obs_positions=obs_pos,
                    obs_velocities=obs_vel,
                    obs_accelerations=obs_acc,
                    a_s=cfg.a_s,
                )

            elapsed = time.perf_counter() - loop_start
            stats_calculator.update(
                out=out,
                trajectory_cart_err=trajectory_cart_err,
                s_index=s_index,
                elapsed_time=elapsed,
                unfeasible_string=unfeasible_status,
                end_eff_pos=end_eff_pos,
            )

            # F. Asynchronous Logging
            if log_data and not stop_event.is_set():
                log_publishers["target"].publish_once(t, nominal_q, nominal_Dq, nominal_DDq)
                log_publishers["state"].publish_once(t, q, dq, ddq)
                log_publishers["cbf"].publish_once(
                    t,
                    [
                        out["h_min"],
                        out["d_min"],
                        out["trajectory_error"],
                        end_eff_pos[0], end_eff_pos[1], end_eff_pos[2],
                        end_eff_vel[0], end_eff_vel[1], end_eff_vel[2],
                        out["vr_min"], out["vh_min"],
                        out["Dtrajectory_time"],
                    ],
                )

            # G. Meshcat Background Visual Update
            if show_data and renderer is not None:
                hud_str = (
                    f"h={out['h_min']:.2f}m  scale={out['Dtrajectory_time']:.3f}  "
                    f"err={out['trajectory_error']:.2f}rad  state:{unfeasible_status}"
                )
                renderer.push_state(
                    out["q"],
                    out["Tbt_nominal"],
                    out["obs_pos"],
                    obstacle_velocities=obs_vel,
                    viz_string=hud_str,
                )

            # H. Time Step & Real-Time Sync
            t += control_period
            if stats_calculator.cycles % 2500 == 0:
                print(f"[Sim Time: {t:6.2f}s] scale={out['Dtrajectory_time']:.3f} | h_min={out['h_min']:6.3f}m | state={unfeasible_status}")

            elapsed_total = time.perf_counter() - loop_start
            rest = control_period - elapsed_total
            if rest > 0:
                time.sleep(rest)

    except KeyboardInterrupt:
        stop_event.set()
        print("\nSimulation stopped by user.")
    finally:
        if log_data and "start" in log_publishers:
            try:
                log_publishers["start"].publish_once(False)
            except Exception:
                pass

    # 8. Print Results
    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE STATISTICS")
    print("=" * 70)
    print(stats_calculator)

    if save_data:
        file_path = str(logging_cfg.get("summary_csv", "../results/simulation_data.csv"))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        headers = [
            "test_type", "lambda_pos", "lambda_vel", "lambda_scaling", "lambda_acc",
            "delta", "gamma", "on_target_rate", "lap_count", "viol_rate",
            "mean_scale", "mean_trajectory_error", "low_scale_rate"
        ]
        delta_deg = float(config.get("controller", {}).get("delta_deg", 4.5))
        final_stats = stats_calculator._calculate_stats()
        row_data = {
            "test_type": test_name,
            "lambda_pos": cfg.lambda_pos,
            "lambda_vel": cfg.lambda_vel,
            "lambda_scaling": cfg.lambda_scaling,
            "lambda_acc": cfg.lambda_acc,
            "delta": delta_deg,
            "gamma": cfg.gamma,
            "on_target_rate": final_stats.get("on_target_rate", 0),
            "lap_count": final_stats.get("lap_count", 0),
            "viol_rate": final_stats.get("violation_rate", 0),
            "mean_scale": final_stats.get("mean_scaling", 0),
            "mean_trajectory_error": final_stats.get("mean_trajectory_error", 0),
            "low_scale_rate": final_stats.get("low_scale_rate", 0),
        }
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
        print(f"Summary results appended to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optimal CBF Controller Simulation / Live Test.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: config/optimal_cbf_params.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)


