"""
Standard robot kinematic waypoints, path planning utilities, and homing helpers.
"""

from __future__ import annotations
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import pinocchio as pin

from cbf_python.trajectory.joint_interpolator import SegmentedJointTrap
from cbf_python.controllers.base_optimal_controller import ControllerConfig
from cbf_python.utils.config_loader import resolve_path


# Standard UR10e Joint Names
UR10E_JOINTS: List[str] = [
    "ur10e_shoulder_pan_joint",
    "ur10e_shoulder_lift_joint",
    "ur10e_elbow_joint",
    "ur10e_wrist_1_joint",
    "ur10e_wrist_2_joint",
    "ur10e_wrist_3_joint",
]

# Standard Home Configuration (radians)
HOME: np.ndarray = np.array([90.0, -140.0, 140.0, -90.0, 90.0, 0.0]) * np.pi / 180.0

# Standard Test Waypoints (radians)
Q10: np.ndarray = np.array([31.0, -78.0, 115.0, -127.0, 86.0, -32.0]) * np.pi / 180.0
Q20: np.ndarray = np.array([31.0, -83.0, 98.0, -110.0, 86.0, -32.0]) * np.pi / 180.0
Q22: np.ndarray = np.array([40.0, -126.0, 141.0, -100.0, 86.0, 45.0]) * np.pi / 180.0
Q25: np.ndarray = np.array([130.0, -100.0, 125.0, -115.0, 94.0, -20.0]) * np.pi / 180.0
Q30: np.ndarray = np.array([136.0, -60.0, 90.0, -122.0, 90.0, 45.0]) * np.pi / 180.0
Q40: np.ndarray = np.array([134.0, -65.0, 70.0, -90.0, 90.0, 45.0]) * np.pi / 180.0


def compute_ee_pose(
    q: np.ndarray,
    model: pin.Model,
    data: pin.Data,
    ee_frame_id: int,
) -> Tuple[np.ndarray, np.ndarray, pin.SE3]:
    """Compute forward kinematics for configuration q and return (position, rotation_matrix, SE3)."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    T_ee = data.oMf[ee_frame_id]
    p = np.array(T_ee.translation)
    R = np.array(T_ee.rotation)
    return p, R, T_ee


def plan_path(planner: SegmentedJointTrap, q_start: np.ndarray) -> None:
    """Populate SegmentedJointTrap with standard demonstration benchmark waypoints."""
    planner.addWayPoint(q_start)
    planner.addWayPoint(Q10)
    planner.addWayPoint(Q20)
    planner.addWayPoint(Q10)
    planner.addWayPoint(Q22)
    planner.addWayPoint(Q25)
    planner.addWayPoint(Q30)
    planner.addWayPoint(Q40)
    planner.addWayPoint(Q30)
    planner.addWayPoint(q_start)


def compute_cartesian_poses(
    q_start: np.ndarray,
    model: pin.Model,
    tool_frame_name: str = "ur10e_wrist_3_joint",
) -> Dict[str, List[float]]:
    """Compute 3D Cartesian coordinates of standard joint waypoints."""
    configs = {
        "q": q_start,
        "q10": Q10,
        "q20": Q20,
        "q22": Q22,
        "q25": Q25,
        "q30": Q30,
        "q40": Q40,
    }
    tool_frame_id = model.getFrameId(tool_frame_name)
    data = model.createData()
    cartesian_configs: Dict[str, List[float]] = {}
    for name, q_val in configs.items():
        p, _, _ = compute_ee_pose(q_val, model, data, tool_frame_id)
        cartesian_configs[name] = p.tolist()
    return cartesian_configs


def bring_robot_home(cfg: ControllerConfig, q_curr: np.ndarray, home: np.ndarray, bridge: Any, ctrl: Any) -> None:
    """Safely execute a joint-space homing trajectory through the bridge and controller."""
    start_planner = SegmentedJointTrap(Dq_max=cfg.Dq_max * 0.25, DDq_max=cfg.DDq_max * 0.25)
    print(f"Bringing robot to home position from {q_curr} to {home}...")
    start_planner.addWayPoint(q_curr)
    start_planner.addWayPoint(home)
    start_time = start_planner.computeTime()
    print(f"Homing trajectory planned: duration = {start_time:.2f} s")
    time.sleep(0.5)

    ctrl.reset_state(q_curr)
    t_init = 0.0
    Tc = cfg.Tc
    traj_time = 0.0

    while np.linalg.norm(home - bridge.getPositions()) > 0.01:
        loop_start = time.perf_counter()
        obs_p, obs_v, obs_a = bridge.getObstacles()
        nominal_q, nominal_Dq, nominal_DDq = start_planner.getMotionLaw(traj_time)

        out = ctrl.step(
            obs_pos=obs_p,
            obs_vel=obs_v,
            obs_acc=obs_a,
            nominal_q=nominal_q,
            nominal_Dq=nominal_Dq,
            nominal_DDq=nominal_DDq,
        )
        q = out["q"]
        bridge.sendCommand(q)

        t_init += Tc
        traj_time = out["trajectory_time"]

        elapsed = time.perf_counter() - loop_start
        rest = Tc - elapsed
        if rest > 0:
            time.sleep(rest)


def create_base_cfg(set_ID: str, Tc: float, filename: str) -> ControllerConfig:
    """Load base controller configuration parameters from an Optuna/parameter search CSV file."""
    csv_file = resolve_path(filename)
    cfg = ControllerConfig(Tc=Tc)
    df = pd.read_csv(csv_file)

    row = df.loc[df["ID"] == set_ID]
    if row.empty:
        raise ValueError(f"Set ID '{set_ID}' not found in {csv_file}")

    cfg.lambda_pos = float(row["lambda_0_pos"].values[0])
    cfg.lambda_vel = float(row["lambda_0_vel"].values[0])
    cfg.lambda_acc = float(row["lambda_0_acc"].values[0])
    cfg.lambda_scaling = float(row["lambda_0_scaling"].values[0])
    cfg.gamma = float(row["gamma_0"].values[0])
    delta = float(row["delta_0_deg"].values[0])

    cfg.delta_q_max[0:2] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta)
    cfg.delta_q_max[2:4] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 2
    cfg.delta_q_max[4:6] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 4
    return cfg

WAYPOINTS: Dict[str, np.ndarray] = {
    "q10": Q10,
    "q20": Q20,
    "q22": Q22,
    "q25": Q25,
    "q30": Q30,
    "q40": Q40,
}
