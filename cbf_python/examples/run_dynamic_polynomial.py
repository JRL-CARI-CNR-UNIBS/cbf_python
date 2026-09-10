"""Dynamic Polynomial CBF Controller runner."""

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
from cbf_python.controllers.polynomial_controller import PolynomialOptimalController, PolynomialControllerConfig
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
from cbf_python.utils.visualizer import VisualizationDaemon, make_summary_figure, plot_lambdas


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
    config = load_run_config("dynamic_polynomial", config_file)
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
    cfg = PolynomialControllerConfig()
    populate_controller_config(cfg, base_defaults)
    if "controller" in config:
        populate_controller_config(cfg, config["controller"])

    delta_deg = float(config.get("controller", {}).get("delta_deg", 4.5))
    cfg.delta_q_max[0:2] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta_deg)
    cfg.delta_q_max[2:4] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta_deg) * 2
    cfg.delta_q_max[4:6] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta_deg) * 4

    ctrl = PolynomialOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True, keypoint_to_log=-1)

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

    q = first_joint_position.copy()
    planner = SegmentedJointTrap(Dq_max=cfg.Dq_max * 0.25, DDq_max=cfg.DDq_max * 0.125)
    plan_path(planner, q)
    T_total = planner.computeTime()
    cartesian_configs = compute_cartesian_poses(q, model)
    stats = StatisticsCalculator(n_wp=9, T_total=T_total, cartesian_configs=cartesian_configs, Tc=cfg.Tc)

    bring_robot_home(cfg, q, HOME, bridge, ctrl)
    ctrl.reset_state(HOME)
    q = HOME.copy()

    t = 0.0
    trajectory_time = 0.0
    computation_times: List[float] = []
    h_log: List[float] = []
    trj_error_log: List[float] = []
    scaling_log: List[float] = []

    gamma_log, lambda_pos_log, lambda_vel_log, lambda_acc_log, lambda_scaling_log = [], [], [], [], []
    time_log = []

    print(f"Starting Dynamic Polynomial CBF Control Loop for duration = {duration}s...")
    while t < duration and not stop_event.is_set():
        loop_start = time.perf_counter()
        obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed=t)
        nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time % T_total)

        out = ctrl.step(
            obs_pos=obs_pos,
            obs_vel=obs_vel,
            obs_acc=obs_acc,
            nominal_q=nominal_q,
            nominal_Dq=nominal_Dq,
            nominal_DDq=nominal_DDq,
        )

        q = out["q"]
        bridge.sendCommand(q)

        end_eff_pos = out["end_effector_pos"]
        end_eff_vel = out["end_effector_vel"]
        trajectory_cart_err = float(np.linalg.norm(end_eff_pos - out["Tbt_nominal"].translation))

        s_index = compute_dynamic_risk_index(
            end_eff_pos=end_eff_pos,
            end_eff_vel=end_eff_vel,
            obs_positions=obs_pos,
            obs_velocities=obs_vel,
            obs_accelerations=obs_acc,
        )

        elapsed = time.perf_counter() - loop_start
        stats.update(out, trajectory_cart_err, s_index, elapsed, out.get("unfeasible", "FEASIBLE"), end_eff_pos)

        h_log.append(out["h_min"])
        trj_error_log.append(out["trajectory_error"])
        scaling_log.append(out["Dtrajectory_time"])
        computation_times.append(elapsed)

        gamma_log.append(cfg.gamma)
        lambda_pos_log.append(cfg.lambda_pos)
        lambda_vel_log.append(cfg.lambda_vel)
        lambda_acc_log.append(cfg.lambda_acc)
        lambda_scaling_log.append(cfg.lambda_scaling)
        time_log.append(t)

        if viz_daemon:
            hud_text = f"Scaling: {out['Dtrajectory_time']:.2f}\nh_min: {out['h_min']:.3f}m"
            viz_daemon.push_state(q=q, Tgoal=out["Tbt_nominal"], obstacles=[p for p in obs_pos], obstacle_velocities=[v for v in obs_vel], viz_string=hud_text)

        t += cfg.Tc
        trajectory_time = out["trajectory_time"]

        rest = cfg.Tc - (time.perf_counter() - loop_start)
        if rest > 0:
            time.sleep(rest)

    print("\n--- Execution Finished ---")
    print(stats)

    if show_data:
        make_summary_figure(computation_times, h_log, trj_error_log, scaling_log, show=True)
        plot_lambdas(time_log, gamma_log, lambda_pos_log, lambda_vel_log, lambda_acc_log, lambda_scaling_log)

    if bridge:
        bridge.shutdown()


if __name__ == "__main__":
    main()
