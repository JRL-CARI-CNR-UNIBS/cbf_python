"""Runs multiple consecutive experiments evaluating polynomial and optimal controllers."""

import os
import time
from typing import Any, Dict, List

import numpy as np
import pinocchio as pin
from sharework import loadSharework

from cbf_python.bridges.fake_bridge import FakeCommandBridge
from cbf_python.controllers.base_optimal_controller import BCFOptimalController, ControllerConfig
from cbf_python.controllers.polynomial_controller import PolynomialOptimalController, PolynomialControllerConfig
from cbf_python.trajectory.joint_interpolator import SegmentedJointTrap
from cbf_python.utils.config_loader import load_run_config, load_yaml, populate_controller_config, resolve_path
from cbf_python.utils.metrics import StatisticsCalculator, compute_dynamic_risk_index
from cbf_python.utils.obstacle_generators import generate_obs_state_h_fixed
from cbf_python.utils.simulation_helpers import (
    HOME,
    UR10E_JOINTS,
    bring_robot_home,
    compute_cartesian_poses,
    compute_ee_pose,
    plan_path,
)


def run_experiment(
    controller_type: str = "polynomial",
    duration: float = 30.0,
    config_file: str = "run.yaml",
) -> Dict[str, float]:
    """Execute a single episode and return calculated metrics."""
    config = load_run_config("dynamic_polynomial", config_file)
    bridge_config = load_yaml("bridges.yaml")

    model_wrapper = loadSharework(UR10E_JOINTS)
    model = model_wrapper.model
    data = model.createData()
    tool_frame_id = model.getFrameId("ur10e_wrist_3_joint")

    base_defaults = load_yaml("controller_defaults.yaml")
    if controller_type == "polynomial":
        cfg = PolynomialControllerConfig()
        populate_controller_config(cfg, base_defaults)
        if "controller" in config:
            populate_controller_config(cfg, config["controller"])
        ctrl = PolynomialOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True)
    else:
        cfg = ControllerConfig()
        populate_controller_config(cfg, base_defaults)
        if "controller" in config:
            populate_controller_config(cfg, config["controller"])
        ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True)

    fake_cfg = bridge_config.get("fake_bridge", {})
    csv_file = resolve_path(fake_cfg.get("csv_path", "skeleton_vectors/skeleton_vectors_23.csv"))
    cam_cfg = fake_cfg.get("camera_pose", {})
    quat = pin.Quaternion(*cam_cfg.get("quaternion", [0.83, 0.185, 0.513, 0.12]))
    quat.normalize()
    R = quat.toRotationMatrix()
    T_wc = pin.SE3(R, np.array(cam_cfg.get("type_0_translation", [-0.094, -0.93, 2.309])))

    bridge = FakeCommandBridge(UR10E_JOINTS, csv_path=str(csv_file), Tworld_to_cam=T_wc)

    q = HOME.copy()
    planner = SegmentedJointTrap(Dq_max=cfg.Dq_max * 0.25, DDq_max=cfg.DDq_max * 0.125)
    plan_path(planner, q)
    T_total = planner.computeTime()
    cartesian_configs = compute_cartesian_poses(q, model)
    stats = StatisticsCalculator(n_wp=9, T_total=T_total, cartesian_configs=cartesian_configs, Tc=cfg.Tc)

    bring_robot_home(cfg, q, HOME, bridge, ctrl)
    ctrl.reset_state(HOME)

    t = 0.0
    trajectory_time = 0.0
    while t < duration:
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

        t += cfg.Tc
        trajectory_time = out["trajectory_time"]

    return stats.calculate_stats()


def main() -> None:
    print("--- Running Sequential Benchmark Experiments ---")
    experiments = [
        ("optimal_base", "run_cbf_optimal.yaml", "optimal"),
        ("polynomial_tuned", "run_dynamic_polynomial.yaml", "polynomial"),
    ]

    for name, cfg_file, c_type in experiments:
        print(f"\n>> Running scenario: {name} (controller: {c_type})")
        res = run_experiment(controller_type=c_type, duration=15.0, config_file=cfg_file)
        print(f"Results for {name}:")
        for k, v in res.items():
            print(f"  {k:30}: {v:.4f}")


if __name__ == "__main__":
    main()
