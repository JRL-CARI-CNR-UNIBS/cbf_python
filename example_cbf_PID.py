# ur10_cbf_main.py
#
# UR10 kinematic simulation with CBF and Meshcat visualization.
# Controller implemented as a class with a .step() method.
# The QP is assembled via two methods:
#   - matrix_ensemble: non-CBF cost (P, b)
#   - cbf_ensemble: CBF inequality constraints (A, c)
import functools
import time
from typing import List

import meshcat.geometry as mgeom
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import math
from sharework import loadSharework

from example_cbf_optimal import _on_sigint_with_bridge
from interpolator import SegmentedSE3Trap
from visualization_daemon import VisualizationDaemon

import signal

from PID_cbf_task_controller import UR10CBFController
from cbf_numba_lib import (
    compute_h,
    range_state_derivative,
    jacobian_psi,
    jacobian_h,
    damped_pinv_svd,
)

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
    p = T_ee.translation          # 3D position
    R = T_ee.rotation             # 3x3 rotation matrix
    return p, R, T_ee

# ---------------------------- CONSTANTS --------------------------------------
C = 0.25   # [m]  minimum separation distance
Tr = 0.15  # [s]  controller-reaction time
a_s = 2.5  # [m/s²] robot decel/accel capability
Tc = 2e-3  # [s]   2 kHz control period
gamma_default = 5.0  # CBF gain
Dq_max: np.ndarray = np.pi * np.array([1, 1, 1, 1, 1, 1], dtype=np.float64) * np.pi

DDq_max: np.ndarray = np.pi * np.array([1, 1, 1, 1, 1, 1], dtype=np.float64) * np.pi * 5.0

def main():
    # --------------------------- MODEL & VISUALS -------------------------- #

    home = np.array([90.0, -140.0, 140.0, -90.0, 90.0, 0.0]) * np.pi / 180.0


    UR10E_JOINTS = [
        "ur10e_shoulder_pan_joint",
        "ur10e_shoulder_lift_joint",
        "ur10e_elbow_joint",
        "ur10e_wrist_1_joint",
        "ur10e_wrist_2_joint",
        "ur10e_wrist_3_joint",
    ]



    model_wrapper = loadSharework(UR10E_JOINTS)
    prefix = 'ur10e_'
    model = model_wrapper.model
    joint_ids = {}
    frame_ids = []
    for name in [ "ur10e_elbow_joint", "ur10e_wrist_3_joint",]:
        # --- Joint ID ---
        jid = model.getJointId(name)
        joint_ids[name] = jid

        # --- Frame ID (if a frame with that name exists) ---
        try:
            fid = model.getFrameId(name)
        except Exception:
            fid = None  # no frame with exactly that name
        frame_ids.append(fid)

    viz = MeshcatVisualizer(model, model_wrapper.collision_model, model_wrapper.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    # ------- BRIDGE SETUP -----------
    USE_BRIDGE = False
    target_name = "ur10e_wrist_3_joint"
    idx = UR10E_JOINTS.index(target_name)


    if USE_BRIDGE:
        from joint_command_bridge_modified import JointStateCommandBridge
        bridge = JointStateCommandBridge(
            ordered_joint_names=UR10E_JOINTS,
            threshold=1.1)  # radians (or native units)
        first_joint_position = bridge.wait_for_first_state( target_name, timeout=5.0)
        signal.signal(signal.SIGINT,
                  functools.partial(_on_sigint_with_bridge, bridge))
        if math.isnan(first_joint_position):
            bridge.shutdown()
            return
        first_joint_position = bridge.getPositions()
        bridge.switch_to_forward_position_controller_service()
    else:
        from fake_command_bridge import FakeCommandBridge
        # Build camera pose from your INITI snippet
        quat = pin.Quaternion(0.83, 0.185, 0.513, 0.12)
        quat.normalize()

        R = quat.toRotationMatrix()

        T_wc = pin.SE3(R, np.array([0.094, -0.93, 2.309]))

        bridge = FakeCommandBridge(
            UR10E_JOINTS,
            csv_path="/home/galileo/Desktop/skeleton_vectors_5.csv",
            Tworld_to_cam=T_wc,
            # slowdown_factor=0.1,
            slowdown_factor=1.0,

        )
        #rclpy.init()
        first_joint_position = home



    # Obstacles (red spheres)
    tmp = np.array([-300, 0., 0.])
    obstacle_positions = [tmp.copy() for _ in range(18 * 5)]
    tmp = np.array([0, 0., 0.])
    obstacle_velocities = [tmp.copy() for _ in range(18 * 5)]
    obstacle_accelerations = obstacle_velocities.copy()

    for i, pos in enumerate(obstacle_positions):
        viz.viewer[f"obstacle_{i}"].set_object(
            mgeom.Sphere(0.1), mgeom.MeshLambertMaterial(color=0xFF0000)
        )

    # Goal box (green)
    side = 0.2
    viz.viewer["goal"].set_object(
        mgeom.Box([side, side, side / 10]), mgeom.MeshLambertMaterial(color=0x00FF00)
    )

    renderer = VisualizationDaemon(viz)  # default 60 Hz

    # ------------------------ CONTROL INITIALISATION ---------------------- #
    data = model.createData()

    q = first_joint_position.copy()
    q10 = np.array([31.0, -78.0, 115.0, -127.0, 86.0, -32.0]) * np.pi / 180.0
    q20 = np.array([31.0, -83.0, 98.0, -110.0, 86.0, -32.0]) * np.pi / 180.0
    q22 = np.array([40.0, -126.0, 141.0, -100.0, 86.0, 45.0]) * np.pi / 180.0
    q25 = np.array([130.0, -100.0, 125.0, -115.0, 94.0, -20.0]) * np.pi / 180.0
    q30 = np.array([136.0, -60.0, 90.0, -122.0, 90.0, 45.0]) * np.pi / 180.0
    q40 = np.array([134.0, -65.0, 70.0, -90.0, 90.0, 45.0]) * np.pi / 180.0

    configs = {
        "q": q,
        "q10": q10,
        "q20": q20,
        "q22": q22,
        "q25": q25,
        "q30": q30,
        "q40": q40,
    }
    ordered_configs = []
    for i in range(3):
        ordered_configs.extend(["q", "q10", "q20", "q10", "q22", "q25", "q30", "q40", "q30", "q"])
    tool_frame_name = target_name

    # Gains (same as original)
    wn = 100.0
    xi = 0.1
    Kp_tra = np.array([1.0, 1.0, 1.0]) * wn ** 2
    Kd_tra = np.array([1.0, 1.0, 1.0]) * 2.0 * xi * wn
    Kp_rot = np.array([1.0, 1.0, 1.0]) * wn ** 2
    Kd_rot = np.array([1.0, 1.0, 1.0]) * 2.0 * xi * wn

    # Controller instance (like 0 in your joint-space example)
    ctrl = UR10CBFController(
        model=model,
        tool_frame_name=tool_frame_name,
        frames_ids=frame_ids,
        Tc=Tc,
        Kp_tra=Kp_tra,
        Kd_tra=Kd_tra,
        Kp_rot=Kp_rot,
        Kd_rot=Kd_rot,
        gamma=gamma_default,
    )
    ctrl.reset_state(q)
    dq = np.zeros(model.nq)
    # We need initial frame pose for the planner
    pin.framesForwardKinematics(model, data, q)
    tool_frame_id = model.getFrameId(tool_frame_name)
    goal_pose_0 = data.oMf[tool_frame_id].copy()


    # 3) Split into linear and angular parts
    v_lin_max = 26.6586*0.1*0.09  # linear velocity [m/s]
    w_max = (44.1351 *0.1*0.09) # angular velocity [rad/s]

    a_lin_max = 650*0.1*0.1  # linear acceleration [m/s^2]
    alpha_max = 750 *0.1*0.1 # angular acceleration [rad/s^2]

    print(f"v_lin_max: {v_lin_max}")
    print(f"w_max: {w_max}")
    print(f"a_lin_max: {a_lin_max}")
    print(f"alpha_max: {alpha_max}")
    # -------------------------- Trajectory planner ------------------------ #
    planner = SegmentedSE3Trap(vlin_max=v_lin_max, vang_max=w_max,
                               alin_max=a_lin_max, aang_max=alpha_max)

    # def pose_eul(z, y, x, xyz):
    #     R = pin.utils.rotate('z', z) @ pin.utils.rotate('y', y) @ pin.utils.rotate('x', x)
    #     return SE3(R, np.array(xyz))
    #
    # planner.addWayPoint(goal_pose_0 * SE3.Identity())
    # planner.addWayPoint(goal_pose_0 * pose_eul(0.0, 0.0, 0.0, [0.30, 0.00, 0.0]))
    # planner.addWayPoint(goal_pose_0 * pose_eul(math.pi / 4, 0.0, 0.0, [0.30, -0.1, 0.020]))
    # planner.addWayPoint(goal_pose_0 * pose_eul(math.pi / 4, 0.0, -math.pi / 4, [0.3, -0.1, 0.2]))
    # planner.addWayPoint(goal_pose_0 * pose_eul(-math.pi / 4, 0.0, 0.0, [0.30, 0.0, 0.0]))
    # planner.addWayPoint(goal_pose_0 * SE3.Identity())

    for name in ordered_configs:
        p, R, T_ee = compute_ee_pose(configs[name], model, data, tool_frame_id)
        planner.addWayPoint(T_ee)
        # print(f"Configuration {name}:")
        # print("  position [m] = ", p)
        # print("  rotation matrix =\n", R)
        # print("  SE3 object = ", T_ee)
        # print()

    T_total = planner.computeTime()

    renderer.publishPath(planner.publishPath())
    print(f"Total time = {T_total:.3f} s")

    # ------------------------------ MAIN LOOP ---------------------------- #
    try:
        t = 0.0
        trajectory_time = 0.0
        Dtrajectory_time = 1.0
        DDtrajectory_time = 0.0

        while t < 150.0:
            loop_start = time.perf_counter()

            goal_pose, nominal_twist_goal, nominal_goal_dtwist = planner.getMotionLaw(
                trajectory_time % T_total
            )
            obstacle_positions, obstacle_velocities, obstacle_accelerations = bridge.getObstacles()
            # Scale if you ever implement time-scaling; currently D=1, DD=0
            twist_goal = nominal_twist_goal * Dtrajectory_time
            goal_dtwist = (
                nominal_goal_dtwist * Dtrajectory_time ** 2.0
                + nominal_twist_goal * DDtrajectory_time
            )

            # # CBF toggle (10 s on / 10 s off)
            # cbf_enabled = (t % 40.0) < 20.0

            # --------- Controller step (this is the key new API) ------------- #
            out = ctrl.step(
                t=t,
                goal_pose=goal_pose,
                twist_goal=twist_goal,
                goal_dtwist=goal_dtwist,
                obstacle_positions=obstacle_positions,
                obstacle_velocities=obstacle_velocities,
                obstacle_accelerations=obstacle_accelerations,
                cbf_enabled=True,
            )

            q = out["q"]
            h_min = out["h_min"]

            # --------------------------- TIMING & VISUALS ------------------- #
            t += Tc
            trajectory_time += Dtrajectory_time * Tc + 0.5 * DDtrajectory_time * Tc ** 2.0
            Dtrajectory_time += DDtrajectory_time * Tc

            elapsed = time.perf_counter() - loop_start
            rest = Tc - elapsed

            vizualization_string = f"h = {h_min:.2f} m"
            if rest > 0:
                renderer.push_state(
                    q,
                    goal_pose,
                    obstacle_positions,
                    vizualization_string,
                )
                # account for visualization time as well
                elapsed = time.perf_counter() - loop_start
                rest = max(0.0, Tc - elapsed)
                time.sleep(rest)

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")


if __name__ == "__main__":
    main()
