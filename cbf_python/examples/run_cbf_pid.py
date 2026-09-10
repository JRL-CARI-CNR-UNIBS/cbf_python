"""Cartesian PID + CBF controller runner."""

import functools
import math
import signal
import threading
import time
from typing import Any, List, Optional

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import rclpy
from sharework import loadSharework

from cbf_python.bridges.fake_bridge import FakeCommandBridge
from cbf_python.bridges.joint_bridge import JointStateCommandBridge
from cbf_python.controllers.pid_cbf_controller import UR10CBFController
from cbf_python.trajectory.se3_interpolator import SegmentedSE3Trap
from cbf_python.utils.config_loader import load_run_config, load_yaml, resolve_path
from cbf_python.utils.simulation_helpers import (
    HOME,
    UR10E_JOINTS,
    compute_cartesian_poses,
    compute_ee_pose,
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
    config = load_run_config("cbf_pid", config_file)
    bridge_config = load_yaml("bridges.yaml")

    flags = config.get("flags", {})
    show_data = flags.get("show_data", True)
    use_bridge = flags.get("use_bridge", False)
    duration = float(flags.get("duration", 30.0))

    ctrl_cfg = config.get("controller", {})
    Tc = float(ctrl_cfg.get("Tc", 0.002))
    Kp_tra = float(ctrl_cfg.get("Kp_tra", 40.0))
    Kd_tra = float(ctrl_cfg.get("Kd_tra", 12.0))
    Kp_rot = float(ctrl_cfg.get("Kp_rot", 20.0))
    Kd_rot = float(ctrl_cfg.get("Kd_rot", 8.0))
    gamma = float(ctrl_cfg.get("gamma", 5.0))
    C = float(ctrl_cfg.get("C", 0.25))
    Tr = float(ctrl_cfg.get("Tr", 0.15))
    a_s = float(ctrl_cfg.get("a_s", 2.5))

    model_wrapper = loadSharework(UR10E_JOINTS)
    model = model_wrapper.model
    data = model.createData()

    target_name = bridge_config.get("joint_bridge", {}).get("target_name", "ur10e_wrist_3_joint")
    tool_frame_id = model.getFrameId(target_name)

    ctrl = UR10CBFController(
        model=model,
        tool_frame_name=target_name,
        Kp_tra=Kp_tra,
        Kd_tra=Kd_tra,
        Kp_rot=Kp_rot,
        Kd_rot=Kd_rot,
        useCbf=True,
        gamma=gamma,
        C=C,
        Tr=Tr,
        a_s=a_s,
        Tc=Tc,
    )

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
        trans = np.array(cam_cfg.get("type_0_translation", [-0.094, -0.93, 2.309]))
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
    ctrl.reset_state(q)

    # Build Cartesian SE3 Trajectory
    _, R_home, T_home = compute_ee_pose(HOME, model, data, tool_frame_id)
    se3_planner = SegmentedSE3Trap(v_max=0.3, a_max=0.5, omega_max=0.5, alpha_max=1.0)
    se3_planner.addWayPoint(T_home)

    cartesian_poses = compute_cartesian_poses(HOME, model)
    for p in cartesian_poses.values():
        T_wp = pin.SE3(R_home, np.asarray(p))
        se3_planner.addWayPoint(T_wp)
    se3_planner.addWayPoint(T_home)
    T_total = se3_planner.computeTime()

    t = 0.0
    computation_times: List[float] = []
    h_log: List[float] = []
    trj_error_log: List[float] = []
    scaling_log: List[float] = []

    print(f"Starting Cartesian PID + CBF Control Loop for duration = {duration}s...")
    while t < duration and not stop_event.is_set():
        loop_start = time.perf_counter()
        obs_pos, obs_vel, obs_acc = bridge.getObstacles(elapsed=t)
        goal_pose, twist_goal, _ = se3_planner.getMotionLaw(t % T_total)

        out = ctrl.step(
            goal_pose=goal_pose,
            twist_goal=twist_goal,
            obstacle_positions=obs_pos,
            obstacle_velocities=obs_vel,
            obstacle_accelerations=obs_acc,
            gamma=gamma,
        )

        q = out["q"]
        bridge.sendCommand(q)

        elapsed = time.perf_counter() - loop_start
        h_log.append(out["h_min"])
        trj_error_log.append(out["trajectory_error"])
        scaling_log.append(1.0)
        computation_times.append(elapsed)

        if viz_daemon:
            hud_text = f"PID Mode\nh_min: {out['h_min']:.3f}m"
            viz_daemon.push_state(q=q, Tgoal=goal_pose, obstacles=[p for p in obs_pos], obstacle_velocities=[v for v in obs_vel], viz_string=hud_text)

        t += Tc
        rest = Tc - (time.perf_counter() - loop_start)
        if rest > 0:
            time.sleep(rest)

    print("\n--- Execution Finished ---")
    if show_data:
        make_summary_figure(computation_times, h_log, trj_error_log, scaling_log, show=True)

    if bridge:
        bridge.shutdown()


if __name__ == "__main__":
    main()
