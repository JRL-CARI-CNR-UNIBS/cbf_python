from typing import Dict, List, Tuple, Sequence
import time
import numpy as np
import pinocchio as pin

from scripts.util.joint_interpolator import SegmentedJointTrap
from Controller.optimal_cbf_task_controller import ControllerConfig, BCFOptimalController

# --------------------------- TEST WAYPOINTS ------------------------------
q10 = np.array([31.0, -78.0, 115.0, -127.0, 86.0, -32.0]) * np.pi / 180.0
q20 = np.array([31.0, -83.0, 98.0, -110.0, 86.0, -32.0]) * np.pi / 180.0
q22 = np.array([40.0, -126.0, 141.0, -100.0, 86.0, 45.0]) * np.pi / 180.0
q25 = np.array([130.0, -100.0, 125.0, -115.0, 94.0, -20.0]) * np.pi / 180.0
q30 = np.array([136.0, -60.0, 90.0, -122.0, 90.0, 45.0]) * np.pi / 180.0
q40 = np.array([134.0, -65.0, 70.0, -90.0, 90.0, 45.0]) * np.pi / 180.0


def generate_velocity(start_point: Sequence[float], end_point: Sequence[float], magnitude: float) -> np.ndarray:
    """Calculates velocity vector [vx, vy, vz] directed from start_point towards end_point."""
    direction = np.asarray(start_point) - np.asarray(end_point)
    distance = np.linalg.norm(direction)
    if distance < 1e-9:
        return np.zeros(3, dtype=float)
    return (direction / distance) * magnitude


def compute_ee_pose(q: np.ndarray, model: pin.Model, data: pin.Data, ee_frame_id: int) -> Tuple[np.ndarray, np.ndarray, pin.SE3]:
    """
    Compute forward kinematics for the end-effector at configuration q.

    Returns:
        position        : np.ndarray (3,)
        rotation_matrix : np.ndarray (3, 3)
        SE3_placement   : pin.SE3
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    T_ee = data.oMf[ee_frame_id]
    return T_ee.translation.copy(), T_ee.rotation.copy(), T_ee.copy()


def bring_robot_home(cfg: ControllerConfig, q: np.ndarray, home: np.ndarray, bridge, ctrl: BCFOptimalController) -> None:
    """Safely transitions the robot from configuration q to home position."""
    start_planner = SegmentedJointTrap(Dq_max=cfg.Dq_max * 0.25, DDq_max=cfg.DDq_max * 0.25)
    print(f"Bringing robot to home position from {q.T} to {home.T}")
    start_planner.addWayPoint(q)
    start_planner.addWayPoint(home)
    Tc = cfg.Tc
    trajectory_time_initial = 0.0
    start_time = start_planner.computeTime()
    print(f"Bringing robot to home position, planned duration: {start_time:.2f} s")
    time.sleep(1.0)
    ctrl.reset_state(q)

    while np.linalg.norm(home - bridge.getPositions()) > 0.01:
        loop_start = time.perf_counter()
        obstacle_positions, obstacle_velocities, obstacle_accelerations = bridge.getObstacles()
        nominal_q, nominal_Dq, nominal_DDq = start_planner.getMotionLaw(trajectory_time_initial)

        out = ctrl.step(
            obs_pos=obstacle_positions,
            obs_vel=obstacle_velocities,
            obs_acc=obstacle_accelerations,
            nominal_q=nominal_q,
            nominal_Dq=nominal_Dq,
            nominal_DDq=nominal_DDq
        )
        q = out["q"]
        bridge.sendCommand(q)

        trajectory_time_initial = out["trajectory_time"]
        elapsed = time.perf_counter() - loop_start
        rest = Tc - elapsed
        if rest > 0:
            time.sleep(rest)


def plan_path(planner: SegmentedJointTrap, q: np.ndarray) -> None:
    """Populates waypoints for standard multi-waypoint cyclic joint trajectory."""
    planner.addWayPoint(q)
    planner.addWayPoint(q10)
    planner.addWayPoint(q20)
    planner.addWayPoint(q10)
    planner.addWayPoint(q22)
    planner.addWayPoint(q25)
    planner.addWayPoint(q30)
    planner.addWayPoint(q40)
    planner.addWayPoint(q30)
    planner.addWayPoint(q)


def compute_cartesian_poses(q: np.ndarray, model: pin.Model) -> Dict[str, np.ndarray]:
    """Computes Cartesian 3D tool positions for all reference joint waypoints."""
    configs = {
        "q": q,
        "q10": q10,
        "q20": q20,
        "q22": q22,
        "q25": q25,
        "q30": q30,
        "q40": q40,
    }
    tool_frame_id = model.getFrameId("ur10e_wrist_3_joint")
    data = model.createData()
    cartesian_configs = {}
    for name, config in configs.items():
        p, _, _ = compute_ee_pose(config, model, data, tool_frame_id)
        cartesian_configs[name] = p
    return cartesian_configs