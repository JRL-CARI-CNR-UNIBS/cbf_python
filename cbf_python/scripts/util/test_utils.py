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
