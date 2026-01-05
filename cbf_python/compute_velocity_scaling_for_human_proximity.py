import numpy as np
import pinocchio as pin


def compute_velocity_scaling_for_human_proximity(
    model: pin.Model,
    data: pin.Data,
    q,
    dq,
    ddq,
    tool_frame_ids,
    human_positions_world,
    minimum_distance: float = 0.25,   # below this distance -> commanded speed must be zero
    reaction_time: float = 0.15,      # [s]
    max_deceleration: float = 2.5,    # [m/s^2]
    human_max_speed: float = 1.6,     # [m/s]
) -> float:
    """
    Compute a global scaling factor in [0, 1] to apply to the robot commanded velocity,
    based on distance and approach speed toward humans.

    For each tool frame (e.g., end-effector points) and for each human position:
      - Compute distance tool->human in WORLD frame
      - Project tool linear velocity onto the tool->human direction
      - If the projected approach speed exceeds the allowed maximum v_max, reduce scaling

    Assumptions:
      - `human_positions_world` are expressed in the WORLD frame.
      - Tool placements `data.oMf[frame_id]` and frame velocities with
        `pin.ReferenceFrame.LOCAL_WORLD_ALIGNED` are consistent with that WORLD frame.
    """

    # Convert configuration and derivatives to 1D float arrays (Pinocchio-friendly).
    q = np.asarray(q, dtype=float).reshape((-1,))
    dq = np.asarray(dq, dtype=float).reshape((-1,))
    ddq = np.asarray(ddq, dtype=float).reshape((-1,))


    # Compute kinematics derivatives and update frame placements (so data.oMf is valid).
    pin.computeForwardKinematicsDerivatives(model, data, q, dq, ddq)
    pin.updateFramePlacements(model, data)

    # Scaling factor: start with no limitation, then tighten with min() across constraints.
    velocity_scaling = 1.0

    # Small epsilon to avoid division by zero when distance is extremely small.
    eps = 1e-12

    # Pre-convert human positions once.
    humans = [np.asarray(p, dtype=float).reshape((-1,)) for p in human_positions_world]

    for frame_id in tool_frame_ids:
        # Tool placement in WORLD frame: oMf = placement of frame f in WORLD (o).
        tool_placement_world = data.oMf[frame_id]
        tool_position_world = np.asarray(tool_placement_world.translation, dtype=float).reshape((-1,))

        # Tool spatial velocity expressed in a world-aligned frame; linear part is in WORLD coordinates.
        tool_twist_world = pin.getFrameVelocity(
            model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        tool_linear_velocity_world = np.asarray(tool_twist_world.linear, dtype=float).reshape((-1,))

        for human_position_world in humans:
            # Relative vector from tool to human (WORLD frame).
            relative_vector = human_position_world - tool_position_world
            distance = float(np.linalg.norm(relative_vector))

            # If too close, enforce full stop immediately.
            if distance < minimum_distance:
                return 0.0

            # Unit direction from tool toward human (WORLD frame).
            direction_to_human = relative_vector / max(distance, eps)

            # Approach speed: projection of tool linear velocity along direction_to_human.
            # Positive => moving toward the human; negative => moving away.
            approach_speed = float(np.dot(tool_linear_velocity_world, direction_to_human))

            # If not approaching, no constraint from this human for this frame.
            if approach_speed <= 0.0:
                continue

            # Compute the maximum allowable approach speed (v_max) from your formula.
            # Clamp the radicand to avoid sqrt of a negative number due to numerical issues.
            radicand = (
                human_max_speed ** 2
                + (max_deceleration * reaction_time) ** 2
                - (2.0 * max_deceleration * (minimum_distance - distance))
            )
            radicand = max(radicand, 0.0)

            max_allowed_speed = (radicand ** 0.5) - (max_deceleration * reaction_time) - human_max_speed
            max_allowed_speed = max(max_allowed_speed, 0.0)  # no negative allowed speed

            # If the current approach speed violates the bound, scale down.
            if approach_speed > max_allowed_speed:
                velocity_scaling = min(velocity_scaling, max_allowed_speed / approach_speed)

                # Early exit: cannot be more restrictive than 0.
                if velocity_scaling <= 0.0:
                    return 0.0

    # Ensure the output is within [0, 1].
    return float(np.clip(velocity_scaling, 0.0, 1.0))

