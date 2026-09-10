"""Hardware bridge verification test using sine trajectory on wrist 3."""

import math
import time
from typing import List

import numpy as np
import rclpy

from cbf_python.bridges.joint_bridge import JointStateCommandBridge
from cbf_python.utils.config_loader import load_yaml
from cbf_python.utils.simulation_helpers import UR10E_JOINTS


def main(config_file: str = "bridges.yaml") -> None:
    bridge_config = load_yaml(config_file)
    j_cfg = bridge_config.get("joint_bridge", {})
    threshold = float(j_cfg.get("threshold", 1.0))
    target_name = j_cfg.get("target_name", "ur10e_wrist_3_joint")

    rclpy.init()
    bridge = JointStateCommandBridge(ordered_joint_names=UR10E_JOINTS, threshold=threshold)

    print(f"Waiting for first joint state of '{target_name}'...")
    first_pos = bridge.wait_for_first_state(target_name, timeout=5.0)
    if math.isnan(first_pos):
        print("Timeout waiting for joint states. Exiting.")
        bridge.shutdown()
        return

    bridge.switch_to_forward_position_controller_service()

    amp = 0.3
    freq = 0.2
    center = first_pos
    idx = UR10E_JOINTS.index(target_name)

    print(f"Driving {target_name} with sine: center={center:.3f}, amp={amp}, freq={freq} Hz")
    t_start = time.time()

    try:
        while rclpy.ok():
            t = time.time() - t_start
            wrist3 = center + amp * math.sin(2.0 * math.pi * freq * t)

            q = bridge.getPositions()
            if np.isnan(q).any():
                print("NaN detected in joint positions. Stopping bridge.")
                break

            q[idx] = wrist3
            pos, vel, acc = bridge.getObstacles()
            if len(pos) > 0:
                print(f"Received {len(pos)} obstacles from camera/ZED.")

            try:
                bridge.sendCommand(q)
            except ValueError as e:
                print(f"Threshold warning: {e}")

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        bridge.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
