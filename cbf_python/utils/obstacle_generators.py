"""
Dynamic obstacle generation and positioning utilities for simulation benchmarks.
"""

from __future__ import annotations
from typing import Tuple, List, Any
import numpy as np
import pinocchio as pin

from cbf_python.controllers.kernels.ssm_cbf_acc import (
    dmin_and_jacobian_numba,
    h_and_jacobian_numba,
)
from cbf_python.utils.simulation_helpers import compute_ee_pose


def generate_velocity(
    start_point: Sequence[float] | np.ndarray,
    end_point: Sequence[float] | np.ndarray,
    magnitude: float,
) -> np.ndarray:
    """Calculate 3D velocity vector pointing from start_point towards end_point with given magnitude."""
    A = np.asarray(start_point, dtype=float).reshape(3)
    B = np.asarray(end_point, dtype=float).reshape(3)
    direction = A - B
    dist = float(np.linalg.norm(direction))
    if dist < 1e-12:
        return np.zeros(3, dtype=float)
    return (direction / dist) * float(magnitude)


def generate_d_value(h_ref: float, v_ref: float, Tr: float = 0.15, a_s: float = 2.5, C: float = 0.25) -> float:
    """Iteratively solve for the distance d that yields safety margin h_ref given relative velocity v_ref."""
    d = 0.0
    h = -10.0
    while h < h_ref:
        h, _ = h_and_jacobian_numba(d, -0.1, v_ref, 0.0, Tr, a_s, C, 1e-9)
        d += 0.01
    return d


def compute_required_d(
    h: float, v_r: float, v_h: float, a_h: float, Tr: float = 0.15, a_s: float = 2.5, C: float = 0.25
) -> float:
    """Compute required distance d analytically for target margin h.
    
    Inverts h_and_jacobian_numba:
      - For h >= 0: h = d_min - C  =>  d = h + C - base_dmin
      - For h < 0:  h = (d_min - C) * (1 - Tr * v_r / C)  =>  d = h / (1 - Tr * v_r / C) + C - base_dmin
    """
    base_dmin, _ = dmin_and_jacobian_numba(0.0, v_r, v_h, a_h, Tr, a_s, 1e-9)
    if h < 0.0:
        denom = 1.0 - (Tr * v_r / C)
        return float(h / denom + C - base_dmin)
    return float(h + C - base_dmin)


def generate_pos_sphere(
    d: float,
    ee_x: float,
    ee_y: float,
    ee_z: float,
    model: pin.Model,
    data: pin.Data,
    ee_vel: np.ndarray,
) -> np.ndarray:
    """Generate obstacle position on lower sphere or cone around the end effector."""
    v_lin = np.asarray(ee_vel, dtype=float).reshape(3)
    v_norm = float(np.linalg.norm(v_lin))

    if v_norm < 1e-6:
        phi = np.random.uniform(0.0, 2.0 * np.pi)
        costheta = np.random.uniform(-1.0, 0.0)
        sintheta = np.sqrt(1.0 - costheta ** 2)
        x = ee_x + d * sintheta * np.cos(phi)
        y = ee_y + d * sintheta * np.sin(phi)
        z = ee_z + d * costheta
        return np.array([[x, y, z]], dtype=float)

    z_vec = v_lin / v_norm
    global_up = np.array([0.0, 0.0, 1.0])
    y_vec = np.cross(global_up, z_vec)

    if np.linalg.norm(y_vec) < 1e-6:
        y_vec = np.array([0.0, 1.0, 0.0])
        x_vec = np.array([1.0, 0.0, 0.0])
    else:
        y_vec = y_vec / np.linalg.norm(y_vec)
        x_vec = np.cross(y_vec, z_vec)

    R = np.column_stack((x_vec, y_vec, z_vec))
    cos_half_angle = np.cos(np.deg2rad(22.5))
    phi = np.random.uniform(-np.pi / 2, np.pi / 2)
    costheta = np.random.uniform(cos_half_angle, 1.0)
    sintheta = np.sqrt(1.0 - costheta ** 2)

    p_local = np.array([d * sintheta * np.cos(phi), d * sintheta * np.sin(phi), d * costheta])
    p_rot = R @ p_local
    x = p_rot[0] + ee_x
    y = p_rot[1] + ee_y
    z = p_rot[2] + ee_z
    return np.array([[x, y, z]], dtype=float)


def generate_pos_v_dir(d: float, ee_x: float, ee_y: float, ee_z: float, v_r: np.ndarray) -> np.ndarray:
    """Generate obstacle position displaced by distance d along velocity direction v_r."""
    center = np.array([ee_x, ee_y, ee_z], dtype=float)
    v_arr = np.asarray(v_r, dtype=float).reshape(3)
    norm_v = float(np.linalg.norm(v_arr))
    if norm_v < 1e-6:
        return center.reshape(1, 3)
    return (center + (v_arr / norm_v) * d).reshape(1, 3)


def generate_obs_state(
    obstacle_positions: np.ndarray,
    obstacle_velocities: np.ndarray,
    cycles: int,
    enable_spawn: bool,
    planner: Any,
    trajectory_time: float,
    T_total: float,
    model: pin.Model,
    data: pin.Data,
    tool_frame_id: int,
    end_eff_pos: np.ndarray,
    Dtrajectory_time: float,
    count_move: int,
) -> Tuple[np.ndarray, np.ndarray, bool, int]:
    """Generate and advance obstacle states along future planned path."""
    if (cycles % 500 == 0) and enable_spawn:
        q_temp, _, _ = planner.getMotionLaw((trajectory_time + 1.0) % T_total)
        obs_p, _, _ = compute_ee_pose(q_temp, model, data, tool_frame_id)
        obs_p = obs_p.copy()
        obs_p[2] -= 0.1
        obstacle_positions = obs_p.reshape(1, 3)
        obs_v = generate_velocity(end_eff_pos, obstacle_positions, 0.1)
        obstacle_velocities = obs_v.reshape(1, 3)
        count_move = 0

    if Dtrajectory_time < 0.05:
        obstacle_positions[0][0] += 0.0015
        obstacle_positions[0][1] += 0.0015
        obstacle_positions[0][2] -= 0.0015
        enable_spawn = False
        count_move += 1
    else:
        enable_spawn = True

    return obstacle_positions, obstacle_velocities, enable_spawn, count_move


def generate_obs_state_h_fixed(
    obstacle_positions: np.ndarray,
    obstacle_velocities: np.ndarray,
    cycles: int,
    enable_spawn: bool,
    model: pin.Model,
    data: pin.Data,
    tool_frame_id: int,
    end_eff_pos: np.ndarray,
    Dtrajectory_time: float,
    count_move: int,
    d_objective: float,
    v_ref: float,
    spawn_freq: int,
    ee_vel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool, int]:
    """Spawn obstacle dynamically at fixed distance / margin objective."""
    if (cycles % spawn_freq == 0) and enable_spawn:
        obstacle_positions = generate_pos_sphere(
            d_objective, end_eff_pos[0], end_eff_pos[1], end_eff_pos[2], model, data, ee_vel
        )
        obs_v = generate_velocity(end_eff_pos, obstacle_positions, v_ref)
        obstacle_velocities = np.asarray(obs_v, dtype=float).reshape(1, 3)

    if Dtrajectory_time < 0.05:
        obstacle_positions[0][0] += 0.0015
        obstacle_positions[0][1] += 0.0015
        obstacle_positions[0][2] -= 0.0015
        enable_spawn = False
        count_move += 1
    else:
        enable_spawn = True

    return obstacle_positions, obstacle_velocities, enable_spawn, count_move


def generate_target_h(h_mean: float, h_std: float) -> float:
    """Sample target safety margin from Gaussian distribution."""
    return float(np.random.normal(loc=h_mean, scale=h_std))
