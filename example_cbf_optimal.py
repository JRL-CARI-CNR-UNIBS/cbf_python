# =============================================================================
# UR10 Kinematic Simulation with Pinocchio and Meshcat (threaded visual updates)
# =============================================================================
#
# This version spawns a background **daemon** thread that handles every visual
# operation (robot pose, moving obstacles, goal box, and HUD text).  The main
# 1 kHz control loop therefore never touches Meshcat directly, so its real‑time
# budget is preserved even on modest hardware.
#
# -----------------------------------------------------------------------------
#                      ***  CHANGES IN THIS REVISION  ***
# -----------------------------------------------------------------------------
# • `flush_visuals()` acquires `render_lock` **non‑blocking**; if the previous
#   visual push is still running we skip this frame instead of waiting.  This
#   prevents the control thread from stalling.
# • Completed the main loop, including the CBF/QP branch, joint‑space
#   integration, shared‑state publication, and fixed‑period sleep.
# • Added graceful keyboard‑interrupt handling: Ctrl‑C shuts down cleanly.
# -----------------------------------------------------------------------------

import time

import meshcat.geometry as mgeom

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from joint_interpolator import SegmentedJointTrap
from visualization_daemon import VisualizationDaemon
from sharework import loadSharework

from bcf_utils import make_summary_figure, print_stats_table


from optimal_cbf_task_controller import BCFOptimalController, ControllerConfig

import math


from scipy.linalg import block_diag


def main():
    # --------------------------- MODEL & VISUALS ---------------------------------
    USE_BRIDGE = False



    duration = 40.0

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

    Tc =2e-3
    cfg = ControllerConfig(Tc=Tc)
    cfg.lambda1 = 1.0e2
    cfg.lambda2 = 1.0
    cfg.lambda3 = 1.0e-1
    cfg.lambda4 = 0.0
    ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg)

    target_name = "ur10e_wrist_3_joint"
    idx = UR10E_JOINTS.index(target_name)
    if USE_BRIDGE:
        from joint_command_bridge_modified import JointStateCommandBridge
        bridge = JointStateCommandBridge(
            ordered_joint_names=UR10E_JOINTS,
            threshold=1.1)  # radians (or native units)
        first_joint_position = bridge.wait_for_first_state( target_name, timeout=5.0)
        if math.isnan(first_joint_position):
            bridge.shutdown()
            return
        first_joint_position = bridge.getPositions()
        bridge.switch_to_forward_position_controller_service()
    else:
        from fake_command_bridge import FakeCommandBridge
        # Build camera pose from your INITI snippet
        R = pin.utils.rotate('z', 1.9) @ pin.utils.rotate('x', 1.57)
        T_wc = pin.SE3(R, np.array([-1.85, -0.9, 0.9]))

        bridge = FakeCommandBridge(
            UR10E_JOINTS,
            csv_path="a01_s10_e02_skeleton3D_with_savgol_vel_acc.csv",
            Tworld_to_cam=T_wc,
            slowdown_factor=0.4,
        )

        first_joint_position = home

    model = model_wrapper.model
    viz = MeshcatVisualizer(model, model_wrapper.collision_model, model_wrapper.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    tmp = np.array([-300, 0., 0.])
    obstacle_positions = [tmp.copy() for _ in range(18*5)]
    tmp = np.array([0, 0., 0.])
    obstacle_velocities = [tmp.copy() for _ in range(18*5)]
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

    # HUD text node
    renderer = VisualizationDaemon(viz)  # default 60 Hz

    # --------------------------- CONTROL INITIALISATION --------------------------
    q = first_joint_position.copy()
    q2 = home.copy()
    q2[1] = -np.pi * 0.5
    q2[2] = np.pi * 0.5

    print(f"q={q.T}\nq={q2.T}")

    planner = SegmentedJointTrap(Dq_max=cfg.Dq_max*.3, DDq_max=cfg.DDq_max*.3)

    # 2 · add way‑points -------------------------------------------
    planner.addWayPoint(q)
    planner.addWayPoint(home)
    planner.addWayPoint(q2)
    planner.addWayPoint(home)

    T_total = planner.computeTime()

    renderer.publishPath(planner.publishPath())

    ct, ct_qp, ct_ssm, ct_planner, ct_pin, h_log, trj_error_log, scaling_log = [], [], [], [], [], [], [], []

    # ------------------------------ MAIN LOOP -------------------- ----------------
    try:

        t = 0.0

        trajectory_time = 0.0


        timeout_cycles = cycles =0

        ctrl.reset_state(q)

        while t < duration:
            h_min = np.inf

            loop_start = time.perf_counter()



            obstacle_positions, obstacle_velocities, obstacle_accelerations = bridge.getObstacles()


            cycles += 1

            nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time % T_total)
            out = ctrl.step(
                obs_pos=obstacle_positions,
                obs_vel=obstacle_velocities,
                obs_acc=obstacle_accelerations,
                nominal_q=nominal_q,
                nominal_Dq=nominal_Dq,
                nominal_DDq=nominal_DDq
            )

            q = out["q"]

            if cycles<5:
                print(f"q pln={nominal_q.T}\nq act={q.T}")

            ddq = out["ddq"]
            trajectory_time = out["trajectory_time"]
            Dtrajectory_time = out["Dtrajectory_time"]

            elapsed = time.perf_counter() - loop_start
            ct_qp.append(elapsed)

            # --------------------------- INTEGRATION ----------------------------
            t += Tc

            if USE_BRIDGE:
                bridge.sendCommand(q)

            # ----------------------------- TIMING -------------------------------
            elapsed = time.perf_counter() - loop_start
            if cycles>1:
                ct.append(elapsed)
                scaling_log.append(Dtrajectory_time)
                h_log.append(out["h_min"])
                trj_error_log.append(out["trajectory_error"])

            rest = Tc - elapsed
            if rest > 0:
                vizualization_string =f"h={out['h_min']:.2f}m  scale={out['Dtrajectory_time']:.3f}  err={out['trajectory_error']:.2f}"

                renderer.push_state(out["q"], out["Tbt_nominal"], out["obs_pos"], vizualization_string)
                elapsed = time.perf_counter() - loop_start
                rest = max(0.0,Tc - elapsed)
                time.sleep(rest)
            else:
                timeout_cycles+=1

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")


    # Call with your
    computation_times = np.array(ct)
    scaling_log = np.array(scaling_log)
    h_log = np.array(h_log)
    trj_error_log = np.array(trj_error_log)

    #computation_times_others=computation_times-(computation_times_planner+computation_times_pin+computation_times_qp+computation_times_ssm)
    stats = {
        "computation_times": computation_times,
    }

    print(f"timeout cycles = {timeout_cycles} over {cycles}, percentage = {100.0*timeout_cycles/cycles}, average = {np.mean(computation_times)}")
    print_stats_table(stats)
    _ = make_summary_figure(
        computation_times,
        h_log,
        trj_error_log,
        scaling_log,
    )


if __name__ == "__main__":
    main()
