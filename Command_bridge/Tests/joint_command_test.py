#!/usr/bin/env python3
"""
Example script: switch to `forward_position_controller` using the
`/controller_manager/switch_controller` service, then drive
`ur10e_wrist_3_joint` with a sine wave using JointStateCommandBridge.

- Creates the bridge with UR10e joint order
- Waits for the first joint state (with a short timeout)
- Publishes a sine motion on wrist_3 around its current position
- Uses a **service call** (not CLI) to stop/start controllers at startup
- Catches ValueError if the command exceeds `threshold`

Run:
  python3 example_wrist3_sine.py

At start, the script will call:
  ros2 service call /controller_manager/switch_controller \
    controller_manager_msgs/srv/SwitchController \
    "activate_controllers: []\n     deactivate_controllers: []\n     start_controllers: ['forward_position_controller']\n     stop_controllers: ['scaled_joint_trajectory_controller']\n     strictness: 0\n     start_asap: false\n     activate_asap: false\n     timeout: {sec: 0, nanosec: 0}"
"""

import math
import time
from typing import List

import numpy as np
import rclpy
from controller_manager_msgs.srv import SwitchController

# Import the class from the same directory or your installed package
from Command_bridge.joint_command_bridge import JointStateCommandBridge


UR10E_JOINTS: List[str] = [
    "ur10e_shoulder_pan_joint",
    "ur10e_shoulder_lift_joint",
    "ur10e_elbow_joint",
    "ur10e_wrist_1_joint",
    "ur10e_wrist_2_joint",
    "ur10e_wrist_3_joint",
]

def main():
    rclpy.init()

    bridge = JointStateCommandBridge(
        ordered_joint_names=UR10E_JOINTS,
        threshold=1.0,  # radians (or native units)
    )

    # Wait (briefly) for a first state to center the sine around it
    target_name = "ur10e_wrist_3_joint"
    idx = UR10E_JOINTS.index(target_name)

    first_joint_position = bridge.wait_for_first_state( target_name, timeout=5.0)
    if math.isnan(first_joint_position):
        bridge.shutdown()
        return

    bridge.switch_to_forward_position_controller_service()


    amp = 0.3         # amplitude (rad); keep < threshold to avoid errors
    freq = 0.2        # Hz
    center = first_joint_position
    print(f"Driving {target_name} with sine: center={center:.3f}, amp={amp}, freq={freq} Hz")
    t_start = time.time()

    try:
        while rclpy.ok():
            t = time.time() - t_start
            wrist3 = center + amp * math.sin(2.0 * math.pi * freq * t)

            # Build a command vector based on the latest known positions
            q = bridge.getPositions()
            # if  NaNs shotdown
            if np.isnan(q).any():
                bridge.shutdown()
                return

            q[idx] = wrist3

            pos,vel,acc=bridge.getObstacles()
            
            if len(pos)>0:
                print(f"received {len(pos)} obstacles")

            try:
                bridge.sendCommand(q)
            except ValueError as e:
                # Exceeded threshold — print and continue (you may choose to break)
                print(f"Threshold error: {e}")

            time.sleep(0.02)  # ~50 Hz update
    except KeyboardInterrupt:
        pass
    finally:
        bridge.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
