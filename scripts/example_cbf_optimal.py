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
import signal
import threading
import functools
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

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
from scripts.util.mean_visualizer import StochasticCBFVisualizer
from scripts.util.bcf_utils import compute_dynamic_risk_index
from scripts.util.test_utils import (
    bring_robot_home,
    plan_path,
    compute_cartesian_poses,
)
from scripts.util import csv_publishers, test_publish_utils as pub_utils


# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
SIMULATION_DURATION: float = 30.0  # seconds
CONTROL_PERIOD: float = 2e-3       # 500 Hz (0.002 s)

SHOW_DATA: bool = True             # Enable Meshcat 3D visualization
USE_BRIDGE: bool = False           # False for offline CSV playback, True for live ROS 2 bridge
LOG_DATA: bool = False             # Log signals to CSV/ROS 2 topics
SAVE_DATA: bool = False            # Append summary statistics to CSV file

UR10E_JOINTS = [
    "ur10e_shoulder_pan_joint",
    "ur10e_shoulder_lift_joint",
    "ur10e_elbow_joint",
    "ur10e_wrist_1_joint",
    "ur10e_wrist_2_joint",
    "ur10e_wrist_3_joint",
]

HOME_CONFIG = np.array([90.0, -140.0, 140.0, -90.0, 90.0, 0.0]) * np.pi / 180.0
TEST_NAME = "recorded_skeleton_23_optimal_cbf"

stop_event = threading.Event()


def _handle_sigint(bridge, signum, frame):
    """Graceful SIGINT interrupt handler."""
    stop_event.set()
    if bridge is not None:
        try:
            bridge.shutdown()
        except Exception:
            pass


def setup_controller(model_wrapper) -> Tuple[ControllerConfig, BCFOptimalController]:
    """Initializes ControllerConfig and BCFOptimalController with tuned parameters."""
    cfg = ControllerConfig(Tc=CONTROL_PERIOD)
    delta_deg = 4.5
    cfg.gamma = 5.95
    cfg.lambda_pos = 2098.0
    cfg.lambda_vel = 0.343
    cfg.lambda_scaling = 16.56
    cfg.lambda_acc = 1.45e-10

    cfg.delta_q_max[0:2] = np.deg2rad(np.ones(2) * delta_deg)
    cfg.delta_q_max[2:4] = np.deg2rad(np.ones(2) * delta_deg * 2.0)
    cfg.delta_q_max[4:6] = np.deg2rad(np.ones(2) * delta_deg * 4.0)

    ctrl = BCFOptimalController(
        model_wrapper=model_wrapper,
        cfg=cfg,
        useCbf=True,
        keypoint_to_log=-1,
    )
    return cfg, ctrl


def setup_visualization(model_wrapper, n_obstacles: int = 18 * 5):
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


def main():
    print("=" * 70)
    print("Starting B-CBF Optimal Controller Execution")
    print(f"Mode: {'ROS 2 Live Bridge' if USE_BRIDGE else 'Offline Simulation Replay'}")
    print(f"Duration: {SIMULATION_DURATION} s | Control Period: {CONTROL_PERIOD} s")
    print("=" * 70)

    # 1. Load Robot Kinematic & Dynamic Model
    model_wrapper = loadSharework(UR10E_JOINTS)
    model = model_wrapper.model

    # 2. Configure Controller
    cfg, ctrl = setup_controller(model_wrapper)
    print(cfg)

    # 3. Setup Command Bridge
    target_name = "ur10e_wrist_3_joint"
    if USE_BRIDGE:
        bridge = JointStateCommandBridge(
            ordered_joint_names=UR10E_JOINTS,
            threshold=1.1,
        )
        first_joint_position = bridge.wait_for_first_state(target_name, timeout=5.0)
        signal.signal(signal.SIGINT, functools.partial(_handle_sigint, bridge))
        if math.isnan(first_joint_position):
            bridge.shutdown()
            return
        first_joint_position = bridge.getPositions()
        bridge.switch_to_forward_position_controller_service()
    else:
        # Camera transformation for recorded human dataset
        quat = pin.Quaternion(0.83, 0.185, 0.513, 0.12)
        quat.normalize()
        T_wc = pin.SE3(quat.toRotationMatrix(), np.array([-0.094, -0.93, 2.309]))

        csv_path = "../skeleton_vectors/skeleton_vectors_23.csv"
        bridge = FakeCommandBridge(
            UR10E_JOINTS,
            csv_path=csv_path,
            Tworld_to_cam=T_wc,
            slowdown_factor=1.0,
            t0=0.0,
        )
        first_joint_position = HOME_CONFIG.copy()
        signal.signal(signal.SIGINT, functools.partial(_handle_sigint, None))

    # 4. Setup Visualization
    renderer = None
    if SHOW_DATA:
        _, renderer = setup_visualization(model_wrapper, n_obstacles=cfg.max_obstacles)

    # 5. Bring Robot to Home Position & Plan Reference Trajectory
    q = first_joint_position.copy()
    if USE_BRIDGE:
        bring_robot_home(cfg, q, HOME_CONFIG, bridge, ctrl)
        q = HOME_CONFIG.copy()

    planner = SegmentedJointTrap(Dq_max=cfg.Dq_max * 0.25, DDq_max=cfg.DDq_max * 0.125)
    plan_path(planner, q)
    T_total = planner.computeTime()
    print(f"Reference trajectory planned: {T_total:.2f} s per cycle")

    if SHOW_DATA and renderer is not None:
        renderer.publishPath(planner.publishPath())

    cartesian_configs = compute_cartesian_poses(q, model)
    stats_calculator = StatisticsCalculator(
        n_wp=10,
        T_total=T_total,
        cartesian_configs=cartesian_configs,
        Tc=CONTROL_PERIOD,
        scaling_threshold=0.5,
    )
    visualizer = StochasticCBFVisualizer()

    # 6. Setup Logging if requested
    log_publishers = {}
    if LOG_DATA:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_path = f"../results/simulation/scaling/{now_str}"
        os.makedirs(test_path, exist_ok=True)
        log_publishers["target"] = csv_publishers.JointTargetCsvPublisher(
            csv_path=f"{test_path}/reference_trajectory.csv",
            column_names="time," + ",".join([f"target_joint_{i}_{attr}" for i in range(6) for attr in ["pos", "vel", "acceleration"]]),
            joint_names=UR10E_JOINTS,
        )
        log_publishers["state"] = csv_publishers.JointTargetCsvPublisher(
            csv_path=f"{test_path}/joint_states.csv",
            column_names="time," + ",".join([f"joint_{i}_{attr}" for i in range(6) for attr in ["pos", "vel", "acceleration"]]),
            joint_names=UR10E_JOINTS,
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
        while t < SIMULATION_DURATION and not stop_event.is_set():
            loop_start = time.perf_counter()

            # A. Get Obstacles
            if USE_BRIDGE:
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
            if USE_BRIDGE and not stop_event.is_set():
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
            visualizer.update_vectors(out["h_min"], out["d_min"], out["vr_min"] - out["vh_min"], t)

            # F. Asynchronous Logging
            if LOG_DATA and not stop_event.is_set():
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
            if SHOW_DATA and renderer is not None:
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
            t += CONTROL_PERIOD
            if stats_calculator.cycles % 2500 == 0:
                print(f"[Sim Time: {t:6.2f}s] scale={out['Dtrajectory_time']:.3f} | h_min={out['h_min']:6.3f}m | state={unfeasible_status}")

            elapsed_total = time.perf_counter() - loop_start
            rest = CONTROL_PERIOD - elapsed_total
            if rest > 0:
                time.sleep(rest)

    except KeyboardInterrupt:
        stop_event.set()
        print("\nSimulation stopped by user.")
    finally:
        if LOG_DATA and "start" in log_publishers:
            try:
                log_publishers["start"].publish_once(False)
            except Exception:
                pass

    # 8. Print Results
    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE STATISTICS")
    print("=" * 70)
    print(stats_calculator)
    visualizer.compute_mean_cov(print_val=True)

    if SAVE_DATA:
        file_path = "../results/simulation_data.csv"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        headers = [
            "test_type", "lambda_pos", "lambda_vel", "lambda_scaling", "lambda_acc",
            "delta", "gamma", "on_target_rate", "lap_count", "viol_rate",
            "mean_scale", "mean_trajectory_error", "low_scale_rate"
        ]
        final_stats = stats_calculator._calculate_stats()
        row_data = {
            "test_type": TEST_NAME,
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
    main()

