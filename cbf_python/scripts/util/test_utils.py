import pinocchio as pin
import numpy as np


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
