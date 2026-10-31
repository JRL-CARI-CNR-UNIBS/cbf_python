"""Gaussian Mixture / Stochastic Parameter CBF Controller runner."""

from datetime import datetime
import functools
import math
import os
import signal
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import rclpy
from sharework import loadSharework

from cbf_python.bridges.fake_bridge import FakeCommandBridge
from cbf_python.bridges.joint_bridge import JointStateCommandBridge
from cbf_python.controllers.gaussian_controller import GaussianController, GaussianControllerConfig
from cbf_python.trajectory.joint_interpolator import SegmentedJointTrap
from cbf_python.utils.config_loader import load_run_config, load_yaml, populate_controller_config, resolve_path
from cbf_python.utils.metrics import StatisticsCalculator, compute_dynamic_risk_index
from cbf_python.utils.publishers import (
    DoubleArrayCsvPublisher,
    DoubleArrayPublisher,
    JointTargetCsvPublisher,
    JointTargetPublisher,
    TestStartCsvPublisher,
    TestStartPublisher,
)
from cbf_python.utils.simulation_helpers import (
    HOME,
    UR10E_JOINTS,
    bring_robot_home,
    compute_cartesian_poses,
    compute_ee_pose,
    plan_path,
)
from cbf_python.utils.visualizer import VisualizationDaemon, make_summary_figure


stop_event = threading.Event()


def _sigint_handler(bridge: Any, signum: int, frame: Any) -> None:
    print("\nInterrupt received! Shutting down gracefully...")
    stop_event.set()
    if bridge is not None:
        try:
            bridge.shutdown()
        except Exception:
            pass


def main(config_file: str = "run.yaml") -> None:
    """Execute Gaussian parameter blending CBF controller simulation or hardware loop."""
    config = load_run_config("gaussian_control", config_file)
    bridge_config = load_yaml("bridges.yaml")

    flags = config.get("flags", {})
    show_data = flags.get("show_data", True)
    use_bridge = flags.get("use_bridge", False)
    log_data = flags.get("log_data", False)
    save_data = flags.get("save_data", False)
    duration = float(flags.get("duration", 30.0))

    params = config.get("parameters", {})
    parameters_type = str(params.get("parameters_type", "0"))
    log_path = str(resolve_path(params.get("log_path", "results/simulation/scaling")))

    model_wrapper = loadSharework(UR10E_JOINTS)
    model = model_wrapper.model
    data = model.createData()

    base_defaults = load_yaml("controller_defaults.yaml")
    cfg = GaussianControllerConfig()
    populate_controller_config(cfg, base_defaults)
    if "controller" in config:
        populate_controller_config(cfg, config["controller"])

    delta_deg = float(config.get("controller", {}).get("delta_deg", 4.5))
    cfg.delta_q_max[0:2] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta_deg)
    cfg.delta_q_max[2:4] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta_deg) * 2
    cfg.delta_q_max[4:6] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta_deg) * 4

    ctrl = GaussianController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True, keypoint_to_log=-1)

    target_name = bridge_config.get("joint_bridge", {}).get("target_name", "ur10e_wrist_3_joint")
    bridge: Any = None

    if use_bridge:
        threshold = float(bridge_config.get("joint_bridge", {}).get("threshold", 1.1))
        bridge = JointStateCommandBridge(ordered_joint_names=UR10E_JOINTS, threshold=threshold)
        first_joint_position = bridge.wait_for_first_state(target_name, timeout=5.0)
        signal.signal(signal.SIGINT, functools.partial(_sigint_handler, bridge))
        if math.isnan(first_joint_position):
            bridge.shutdown()
            return
        first_joint_position = bridge.getPositions()
        bridge.switch_to_forward_position_controller_service()
    else:
        fake_cfg = bridge_config.get("fake_bridge", {})
        csv_file = resolve_path(fake_cfg.get("csv_path", "skeleton_vectors/skeleton_vectors_23.csv"))
        cam_cfg = fake_cfg.get("camera_pose", {})
        quat = pin.Quaternion(*cam_cfg.get("quaternion", [0.83, 0.185, 0.513, 0.12]))
        quat.normalize()
        R = quat.toRotationMatrix()
        trans_key = "type_1_translation" if parameters_type == "1" else "type_0_translation"
        trans = np.array(cam_cfg.get(trans_key, [-0.094, -0.93, 2.309]))
        T_wc = pin.SE3(R, trans)

        bridge = FakeCommandBridge(
            UR10E_JOINTS,
            csv_path=str(csv_file),
            Tworld_to_cam=T_wc,
            slowdown_factor=float(fake_cfg.get("slowdown_factor", 1.0)),
            t0=float(fake_cfg.get("t0", 0.0)),
        )
        if not rclpy.ok():
            rclpy.init()
        first_joint_position = HOME.copy()
        signal.signal(signal.SIGINT, functools.partial(_sigint_handler, bridge))

    viz_daemon: Optional[VisualizationDaemon] = None
    if show_data:
        viz = MeshcatVisualizer(model_wrapper.model, model_wrapper.collision_model, model_wrapper.visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
        viz_daemon = VisualizationDaemon(viz=viz, refresh_hz=60.0)

    stats = StatisticsCalculator()
    interpolator = SegmentedJointTrap(plan_path(first_joint_position), speed=0.3, acceleration=0.2)
    interpolator.setup()

    # Move smoothly to starting position
    bring_robot_home(bridge, first_joint_position, viz_daemon=viz_daemon)
    ctrl.reset_state(first_joint_position)

    dt = cfg.Tc
    t = 0.0

    print("Starting Gaussian Parameter Modulation CBF execution...")
    try:
        while not stop_event.is_set() and t < duration:
            t_start = time.perf_counter()

            # Obstacle sensing
            pos_obs, vel_obs, acc_obs = bridge.getObstacles(elapsed=t)

            # Waypoint reference
            nominal_q, nominal_dq, nominal_ddq = interpolator.evaluate(ctrl.trajectory_time)

            # Controller step
            out = ctrl.step(
                obs_pos=pos_obs,
                obs_vel=vel_obs,
                obs_acc=acc_obs,
                nominal_q=nominal_q,
                nominal_Dq=nominal_dq,
                nominal_DDq=nominal_ddq,
            )

            # Command execution
            bridge.sendCommand(out["q"])

            if viz_daemon is not None:
                viz_daemon.update_robot_and_obstacles(out["q"], pos_obs)

            stats.update(
                t=t,
                h_min=out["h_min"],
                d_min=out["d_min"],
                v_r=out["vr_min"],
                v_h=out["vh_min"],
                scaling=out["Dtrajectory_time"],
                traj_err=out["trajectory_error"],
                unfeasible=(out["unfeasible_cnt"] != "FEASIBLE"),
            )

            t += dt
            elapsed = time.perf_counter() - t_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        if viz_daemon is not None:
            viz_daemon.stop()
        if bridge is not None:
            bridge.shutdown()

    print("\nExecution complete. Performance Metrics:")
    stats.print_summary()


if __name__ == "__main__":
    main()
