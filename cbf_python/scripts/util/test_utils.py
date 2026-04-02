import pinocchio as pin
import numpy as np
import pandas as pd
from Controller.gaussian_controller import GaussianController, GaussianControllerConfig, GaussianSet
import time
from util.joint_interpolator import SegmentedJointTrap
from Controller.optimal_cbf_task_controller import ControllerConfig
# ---------------------------TEST WAYPOINTS ------------------------------
q10 = np.array([31.0, -78.0, 115.0, -127.0, 86.0, -32.0]) * np.pi / 180.0
q20 = np.array([31.0, -83.0, 98.0, -110.0, 86.0, -32.0]) * np.pi / 180.0
q22 = np.array([40.0, -126.0, 141.0, -100.0, 86.0, 45.0]) * np.pi / 180.0
q25 = np.array([130.0, -100.0, 125.0, -115.0, 94.0, -20.0]) * np.pi / 180.0
q30 = np.array([136.0, -60.0, 90.0, -122.0, 90.0, 45.0]) * np.pi / 180.0
q40 = np.array([134.0, -65.0, 70.0, -90.0, 90.0, 45.0]) * np.pi / 180.0


def generate_velocity(start_point, end_point, magnitude):
    """
    Calculates velocity components (vx, vy, vz) from start_point to end_point.

    Args:
        start_point: List or array [x, y, z]
        end_point: List or array [x, y, z]
        magnitude: The desired speed (scalar)

    Returns:
        List of [vx, vy, vz]
    """
    # Convert both to numpy arrays for vector math
    A = np.array(start_point)
    B = np.array(end_point)

    # 1. Get the direction vector
    direction = A - B

    # 2. Calculate the Euclidean distance (norm)
    distance = np.linalg.norm(direction)

    # 3. Handle the case where A and B are the same point to avoid division by zero
    if distance == 0:
        return [0.0, 0.0, 0.0]

    # 4. Normalize and scale
    velocity = (direction / distance) * magnitude

    # Return as a list
    return velocity

def compute_ee_pose(q, model, data, ee_frame_id):
    """
    Compute forward kinematics of the end-effector for joint config q.
    Returns (position, rotation_matrix, SE3).
    """
    # Forward kinematics for all joints
    pin.forwardKinematics(model, data, q)
    # Update frame placements
    pin.updateFramePlacements(model, data)

    T_ee = data.oMf[ee_frame_id]  # SE3 from world (o) to frame (f=tool0)
    p = T_ee.translation  # 3D position
    R = T_ee.rotation  # 3x3 rotation matrix
    return p, R, T_ee


def generate_obs_state(obstacle_positions, obstacle_velocities, cycles, enable_spawm, planner, trajectory_time, T_total, model, data, tool_frame_id, end_eff_pos, Dtrajectory_time, count_move):
    if (cycles % 500 == 0) and enable_spawm:
        q_temp, dq_temp, ddq_temp = planner.getMotionLaw((trajectory_time + 1) % T_total)
        obstacle_positions, a, b = compute_ee_pose(q_temp, model, data, tool_frame_id)
        obstacle_positions = obstacle_positions.tolist()
        obstacle_positions[0] = obstacle_positions[0] + 0.0
        obstacle_positions[1] = obstacle_positions[1] + 0.0
        obstacle_positions[2] = obstacle_positions[2] - 0.1
        obstacle_positions = np.array(obstacle_positions)
        obstacle_positions = obstacle_positions.reshape(1, 3)
        obstacle_velocities = generate_velocity(end_eff_pos, obstacle_positions, 0.1)
        obstacle_velocities = obstacle_velocities.reshape(1, 3)
        # obstacle_accelerations = obstacle_accelerations.reshape(1, 3)
        count_move = 0
        # if Dtrajectory_time < 0.05:
        #     consecutive_low_scale_cycles += 1
        # else:
        #     consecutive_low_scale_cycles = 0
        # if consecutive_low_scale_cycles > 250 and count_move < 25:
        # if  count_move < 25:
    if Dtrajectory_time < 0.05:
        obstacle_positions[0][0] += 0.0015
        obstacle_positions[0][1] += 0.0015
        obstacle_positions[0][2] -= 0.0015
        enable_spawm = False
        count_move += 1
    else:
        enable_spawm = True

    return obstacle_positions, obstacle_velocities, enable_spawm, count_move

def create_base_cfg(set_ID, Tc, filename):
    cfg = ControllerConfig(Tc=Tc)

    df = pd.read_csv(filename)

    cfg.lambda_pos = float(df.loc[df["ID"] == set_ID, f"lambda_0_pos"].values[0])
    cfg.lambda_vel = float(df.loc[df["ID"] == set_ID, f"lambda_0_vel"].values[0])
    cfg.lambda_acc = float(df.loc[df["ID"] == set_ID, f"lambda_0_acc"].values[0])
    cfg.lambda_scaling = float(df.loc[df["ID"] == set_ID, f"lambda_0_scaling"].values[0])
    cfg.gamma = float(df.loc[df["ID"] == set_ID, f"gamma_0"].values[0])
    delta = float(df.loc[df["ID"] == set_ID, f"delta_0_deg"].values[0])

    cfg.delta_q_max[0:2] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta)
    cfg.delta_q_max[2:4] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 2
    cfg.delta_q_max[4:6] = np.deg2rad(np.array([1, 1], dtype=np.float64) * delta) * 4
    return cfg

def bring_robot_home(cfg, q, home, bridge, ctrl):
    start_planner = SegmentedJointTrap(Dq_max=cfg.Dq_max * 0.25, DDq_max=cfg.DDq_max * 0.25)
    print(f"Bringing robot to home position from {q.T} to {home.T}")
    start_planner.addWayPoint(q)
    start_planner.addWayPoint(home)
    t_initial = 0.0
    Tc = cfg.Tc
    trajectory_time_initial = 0.0
    start_time = start_planner.computeTime()
    print(f"Bringing robot to home position, total time: {start_time}")
    time.sleep(1.0)
    ctrl.reset_state(q)
    # test_start = True
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

        # --------------------------- INTEGRATION ----------------------------
        t_initial += Tc
        trajectory_time_initial = out["trajectory_time"]

        elapsed = time.perf_counter() - loop_start

        rest = Tc - elapsed
        if rest > 0:
            rest = max(0.0, Tc - elapsed)
            time.sleep(rest)
def plan_path(planner, q):
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

def compute_cartesian_poses(q, model):
    configs = {
        "q": q,
        "q10": q10,
        "q20": q20,
        "q22": q22,
        "q25": q25,
        "q30": q30,
        "q40": q40,
    }
    cartesian_configs = {
        "q": 0.0,
        "q10": 0.0,
        "q20": 0.0,
        "q22": 0.0,
        "q25": 0.0,
        "q30": 0.0,
        "q40": 0.0,
    }
    tool_frame_name = "ur10e_wrist_3_joint"
    tool_frame_id = model.getFrameId(tool_frame_name)
    data = model.createData()
    for name in cartesian_configs:
        p, R, T_ee = compute_ee_pose(configs[name], model, data, tool_frame_id)
        cartesian_configs[name] = p.tolist()
    return cartesian_configs