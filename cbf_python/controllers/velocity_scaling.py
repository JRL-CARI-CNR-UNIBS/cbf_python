"""
Velocity scaling computation for human proximity and collision prevention.
"""

from __future__ import annotations
from typing import Sequence, Iterable
import numpy as np
import pinocchio as pin


def compute_velocity_scaling_for_human_proximity(
    model: pin.Model,
    data: pin.Data,
    q: Sequence[float] | np.ndarray,
    dq: Sequence[float] | np.ndarray,
    ddq: Sequence[float] | np.ndarray,
    tool_frame_ids: Iterable[int],
    human_positions_world: Iterable[Sequence[float] | np.ndarray],
    minimum_distance: float = 0.25,   # Distance below which commanded speed is zero [m]
    reaction_time: float = 0.15,      # Reaction time [s]
    max_deceleration: float = 2.5,    # Maximum robot deceleration [m/s^2]
    human_max_speed: float = 1.6,     # Assumed maximum human approach speed [m/s]
) -> float:
    """Compute a global velocity scaling factor in [0, 1] based on distance and approach velocity toward humans.

    Parameters
    ----------
    model : pin.Model
        Robot kinematic model.
    data : pin.Data
        Robot kinematic data container.
    q : array-like
        Current joint positions.
    dq : array-like
        Current joint velocities.
    ddq : array-like
        Current joint accelerations.
    tool_frame_ids : iterable of int
        Pinocchio frame IDs to monitor on the robot.
    human_positions_world : iterable of array-like
        Coordinates of obstacles or human keypoints in world frame.
    minimum_distance : float, optional
        Protective separation threshold (default 0.25 m).
    reaction_time : float, optional
        System reaction time delay (default 0.15 s).
    max_deceleration : float, optional
        Robot stopping deceleration limit (default 2.5 m/s^2).
    human_max_speed : float, optional
        Maximum human velocity estimate (default 1.6 m/s).

    Returns
    -------
    float
        Scaling factor in [0, 1].
    """
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    dq_arr = np.asarray(dq, dtype=float).reshape(-1)
    ddq_arr = np.asarray(ddq, dtype=float).reshape(-1)

    pin.computeForwardKinematicsDerivatives(model, data, q_arr, dq_arr, ddq_arr)
    pin.updateFramePlacements(model, data)

    velocity_scaling = 1.0
    eps = 1e-12
    humans = [np.asarray(p, dtype=float).reshape(-1) for p in human_positions_world]

    for frame_id in tool_frame_ids:
        tool_placement_world = data.oMf[frame_id]
        tool_pos = np.asarray(tool_placement_world.translation, dtype=float).reshape(-1)

        tool_twist = pin.getFrameVelocity(
            model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        tool_vlin = np.asarray(tool_twist.linear, dtype=float).reshape(-1)

        for h_pos in humans:
            rel_vec = h_pos - tool_pos
            dist = float(np.linalg.norm(rel_vec))

            if dist < minimum_distance:
                return 0.0

            direction = rel_vec / max(dist, eps)
            approach_speed = float(np.dot(tool_vlin, direction))

            if approach_speed <= 0.0:
                continue

            radicand = (
                human_max_speed ** 2
                + (max_deceleration * reaction_time) ** 2
                - (2.0 * max_deceleration * (minimum_distance - dist))
            )
            radicand = max(radicand, 0.0)

            max_allowed = (radicand ** 0.5) - (max_deceleration * reaction_time) - human_max_speed
            max_allowed = max(max_allowed, 0.0)

            if approach_speed > max_allowed:
                velocity_scaling = min(velocity_scaling, max_allowed / approach_speed)
                if velocity_scaling <= 0.0:
                    return 0.0

    return float(np.clip(velocity_scaling, 0.0, 1.0))
