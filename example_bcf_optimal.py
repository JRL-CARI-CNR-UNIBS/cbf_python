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

import threading
import time
from typing import List

import meshcat.geometry as mgeom
import meshcat.transformations as tf
import meshcat_shapes
import numpy as np
import pinocchio as pin
import quadprog
from example_robot_data import load
from pinocchio.visualize import MeshcatVisualizer

from ssm_cbf_acc import h_and_jacobian_numba, jacobian_psi_times_fg_fast_numba, compute_h_and_constraints_numba
from interpolator import SegmentedSE3Trap
from joint_interpolator import SegmentedJointTrap
from visualization_daemon import VisualizationDaemon
from pinocchio import SE3
from sharework import loadSharework

from human_pose_reader import  PoseReader
from bcf_utils import make_summary_figure, print_stats_table



import math


from scipy.linalg import block_diag


def damped_pinv_svd(J, lam=1e-4):
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_damped = S / (S ** 2 + lam ** 2)  # approssimazione di S^-1
    return (Vt.T * S_damped) @ U.T


def main():
    # --------------------------- MODEL & VISUALS ---------------------------------
    USE_CBF = True
    USE_BRIDGE = False

    # ---------------------------- CONSTANTS --------------------------------------
    C = 0.25  # [m]  minimum separation distance
    Tr = 0.5  # [s]  controller‑reaction time
    a_s = 4.5  # [m/s²] robot decel/accel capability
    Tc = 2e-3  # [s]   2 kHz control period

    delta_q_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])*0.01
    Dq_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])*np.pi*1.0
    DDq_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])*np.pi*3.0
    gamma = 5.0  # CBF gain

    # Dq(k+1)=Dq+DDq*Tc
    # q(k+1)=q+Dq*Tc+0.5*DDq*Tc**2
    # Dtrajectory_time(k+1)= Dtrajectory_time+DDtrajectory_time*Tc

    # ----------------------------- QP SOLVE -----------------------------
    # Goal 1: quadratic error on q(k+1)-nominal_q
    # Goal 2: quadratic error on Dq(k+1)-Dtrajectory_time(k+1)*nominal_Dq
    # Goal 3: quadratic error on Dtrajectory_time(k+1)-1
    # Goal 4: quadratic error on DDq

    # if unfeasible, minimize this problem minizime(0.5*Dq'*Dq+0.5*Dtrajectory_time'*Dtrajectory_time)

    # u =[DDq,DDtrajectory_time]
    #
    # Scaling constraints
    # 0 <  Dtrajectory_time + DDtrajectory_time*Tc < 1
    # Dtrajectory_time + DDtrajectory_time*Tc > 0
    # DDtrajectory_time*Tc > -Dtrajectory_time


    lambda1 = 1.0e2
    lambda2 = 1e0
    lambda3 = 1e-1
    lambda4 = 0e-9
    duration = 30.0
    DDtrajectory_time_max = 1.0

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



    target_name = "ur10e_wrist_3_joint"
    idx = UR10E_JOINTS.index(target_name)
    if USE_BRIDGE:
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
    data = model.createData()
    q = first_joint_position.copy()

    dq = np.zeros(model.nq)
    ddq = np.zeros(model.nq)

    q2 = home.copy()
    q2[1] = -np.pi * 0.5
    q2[2] = np.pi * 0.5

    tool_frame_id = model.getFrameId(prefix+"tool0")
    elbow_frame_id = model.getFrameId(prefix+"forearm_link")
    frames_ids=[elbow_frame_id,tool_frame_id]


    pin.framesForwardKinematics(model, data, q)

    scaling_limit_matrix = np.append(np.zeros(model.nq), Tc)

    planner = SegmentedJointTrap(Dq_max=Dq_max*.3, DDq_max=DDq_max*.3)

    # 2 · add way‑points -------------------------------------------
    planner.addWayPoint(q)
    planner.addWayPoint(home)
    planner.addWayPoint(q2)
    planner.addWayPoint(home)

    T_total = planner.computeTime()

    renderer.publishPath(planner.publishPath())

    I=np.eye(model.nq)

    timeout_cycles=0
    cycles=0
    ct, ct_qp, ct_ssm, ct_planner, ct_pin, h_log, trj_error_log, scaling_log = [], [], [], [], [], [], [], []

    # Goal 1: position error (qn-qn)^2
    P1 = block_diag(0.25 * Tc ** 4 * I, 0)

    # goal 2: velocity error (Dqn*Dtrajectory_time-Dq)^2
    P2_11 = Tc ** 2 * I
    P2 = np.zeros((model.nq + 1, model.nq + 1))
    P2[:model.nq, :model.nq] = P2_11

    # goal 3: scaling error (Dtrajectory_time-1)^2
    P3 = block_diag(0 * I, Tc ** 2)

    P4 = block_diag(I,0)
    b4 = np.array([0] * (model.nq + 1)).flatten()

    Punfeasible = block_diag(Tc**2*I , Tc**2)


    b1 = np.zeros(model.nq + 1)
    b2 = np.zeros(model.nq + 1)
    b3 = np.zeros(model.nq + 1)
    b3[:model.nq] = 0.0  # fixed each loop except last entry
    bunfeasible = np.zeros(model.nq + 1)

    P = np.empty((model.nq + 1, model.nq + 1))
    b = np.empty(model.nq + 1)

    J = np.zeros((6, model.nv))
    dJ = np.zeros((6, model.nv))

    def build_free_forced_one_step(Ts, nq):
        I = np.eye(nq)
        ForcedPos = 0.5 * (Ts ** 2) * I  # (nq x nq)
        FreePos = np.hstack([I, Ts * I])  # (nq x 2nq)
        ForcedVel = Ts * I  # (nq x nq)
        FreeVel = np.hstack([0*I, I])  # (nq x 2nq)
        return FreePos, ForcedPos, FreeVel, ForcedVel

    FreePos, ForcedPos, FreeVel, ForcedVel = build_free_forced_one_step(Tc,model.nq)

    x0 = np.hstack((q, dq))
    # ------------------------------ MAIN LOOP -------------------- ----------------
    try:

        t = 0.0

        trajectory_time = 0.0
        Dtrajectory_time = 1.0
        DDtrajectory_time = 0.0

        max_obstacles = 18*5

        n_constraints = 3+ 2*3*model.nq +max_obstacles*len(frames_ids)
        constraint_matrix = np.zeros((n_constraints, model.nq + 1))
        constraint_vector = np.zeros(n_constraints)
        constraint_matrix[0, :] = -scaling_limit_matrix
        constraint_matrix[1, :] = scaling_limit_matrix
        constraint_matrix[2, -1] = -1
        row_idx=3
        
        # position limits in tube
        constraint_matrix[row_idx:(row_idx + model.nq), 0:model.nq] = -ForcedPos
        row_idx += model.nq
        constraint_matrix[row_idx:(row_idx + model.nq), 0:model.nq] = ForcedPos
        row_idx += model.nq
        
        # velocity limits 
        constraint_matrix[row_idx:(row_idx + model.nq), 0:model.nq] = -ForcedVel
        row_idx += model.nq
        constraint_matrix[row_idx:(row_idx + model.nq), 0:model.nq] = ForcedVel
        row_idx += model.nq

        # acceleration limits
        constraint_matrix[row_idx:(row_idx + model.nq), 0:model.nq] = - np.eye(model.nq)
        row_idx += model.nq
        constraint_matrix[row_idx:(row_idx + model.nq), 0:model.nq] =  np.eye(model.nq)
        row_idx += model.nq
        
        J = np.zeros((6, model.nv))
        dJ = np.zeros((6, model.nv))

        while t < duration:
            h_min = np.inf

            loop_start = time.perf_counter()

            x0=np.hstack((q,dq))


            obstacle_positions, obstacle_velocities, obstacle_accelerations = bridge.getObstacles()


            cycles += 1

            nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time % T_total)

            trajectory_error=np.linalg.norm(q-nominal_q)

            trj_error_log.append(trajectory_error)
            pin.framesForwardKinematics(model, data, nominal_q)
            Tbt_nominal = data.oMf[tool_frame_id].copy()

            elapsed = time.perf_counter() - loop_start
            ct_planner.append(elapsed)

            # ------------------------- CBF QP SETUP -------------------------
            row_idx = 0  # reset index at each loop

            constraint_vector[row_idx] = -(1 - Dtrajectory_time)
            row_idx += 1

            constraint_vector[row_idx] = -Dtrajectory_time
            row_idx += 1

            constraint_vector[row_idx] = -DDtrajectory_time_max
            row_idx += 1

            # tube constraints: nominal_q-delta_q_max < q < nominal_q+delta_q_max
            constraint_vector[row_idx:(row_idx + model.nq)] = -nominal_q - delta_q_max + FreePos @ x0
            row_idx += model.nq
            constraint_vector[row_idx:(row_idx+model.nq)] = nominal_q-delta_q_max - FreePos@x0
            row_idx += model.nq

            # velocity constraints: -Dq_max < Dq < +Dq_max
            # Dq < +Dq_max => -Dq > -Dq_max => -FreeVel@x0 - ForcedVel@u > -Dq_max => -ForcedVel@u > -Dq_max - FreeVel@x0
            constraint_vector[row_idx:(row_idx + model.nq)] = -Dq_max + FreeVel @ x0
            row_idx += model.nq
            # Dq > -Dq_max  => FreeVel@x0 + ForcedVel@u > -Dq_max => ForcedVel@u > -Dq_max - FreeVel@x0
            constraint_vector[row_idx:(row_idx + model.nq)] = -Dq_max - FreeVel @ x0
            row_idx += model.nq
            # acceleration constraints: -DDq_max < DDq < +DDq_max

            # DDq < +DDq_max => -DDq > -DDq_max => -I@u > -DDq_max => -I@u > -DDq_max
            constraint_vector[row_idx:(row_idx + model.nq)] = -DDq_max
            row_idx += model.nq
            # DDq > -DDq_max => I@u > -DDq_max
            constraint_vector[row_idx:(row_idx + model.nq)] =  -DDq_max
            row_idx += model.nq

            t_pin_1 = time.perf_counter()

            pin.computeForwardKinematicsDerivatives(model, data, q, dq, ddq)

            elapsed_pin = time.perf_counter() - t_pin_1
            elapsed_ssm=0.0

            if USE_CBF:
            
                for frame_id in frames_ids:
                    t_pin_1 = time.perf_counter()
                    Tbt = data.oMf[frame_id]
                    translation_bt = Tbt.translation

                    # Current twist
                    twist = pin.getFrameVelocity(
                        model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                    )
                    vel_lineare = twist.linear

                    #
                    J*=0
                    dJ*=0
                    J = pin.computeFrameJacobian(
                        model,
                        data,
                        q,
                        frame_id,
                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    )

                    dJ = pin.frameJacobianTimeVariation(
                        model,
                        data,
                        q,
                        dq,
                        frame_id,
                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                    )
                    Jlin = J[:3, :]
                    dJlin = dJ[:3, :]

                    elapsed_pin += time.perf_counter() - t_pin_1


                    # append constraints on joint position, joint velocity, joint torque
                    t_ssm_1 = time.perf_counter()
                    for i, (obs_pos, obs_vel, obs_acc) in enumerate(zip(obstacle_positions, obstacle_velocities, obstacle_accelerations)):

                        h, row, bound = compute_h_and_constraints_numba(
                            translation_bt, obs_pos, vel_lineare, obs_vel, Tr, a_s, C, obs_acc, 1e-12, Jlin, dJlin, dq, gamma
                        )
                        h_min = min(h_min,h)

                        # Fill preallocated row
                        constraint_matrix[row_idx, :-1] = row
                        constraint_vector[row_idx] = bound
                        row_idx += 1

                elapsed_ssm += min(time.perf_counter() - t_ssm_1,50e-3)
            for i in range(row_idx,n_constraints):
                constraint_matrix[i,:]=0.0
                constraint_vector[i]=-1.0

            h_log.append(h_min)
            ct_ssm.append(elapsed_ssm)
            ct_pin.append(elapsed_pin)


            b1[:-1]=(nominal_q-q-dq*Tc)*0.5*Tc**2
            b3[-1] = -Tc*(Dtrajectory_time-1)

            P2[:model.nq, -1] = -(Tc ** 2) * nominal_Dq
            P2[-1, :model.nq] = P2[:model.nq, -1]
            P2[-1, -1] = Tc ** 2 * nominal_Dq.dot(nominal_Dq)

            b2[:-1] = (nominal_Dq*Dtrajectory_time-dq)*Tc
            b2[-1] = -(nominal_Dq*Dtrajectory_time-dq).dot(nominal_Dq*Tc)



            P=lambda1*P1+lambda2*P2+lambda3*P3+lambda4*P4
            b=lambda1*b1+lambda2*b2+lambda3*b3+lambda4*b4
            t_qp_1 = time.perf_counter()
            try:
                u, *_ = quadprog.solve_qp(
                    P,
                    b,
                    constraint_matrix.T,
                    constraint_vector,
                    0)
                ddq = u[:-1]
                DDtrajectory_time=u[-1]
            except ValueError as err:
                if "constraints are inconsistent" in str(err):

                    bunfeasible[:-1] = -Tc * dq
                    bunfeasible[-1] = -Tc * Dtrajectory_time
                    u, *_ = quadprog.solve_qp(
                        Punfeasible,
                        bunfeasible,
                        constraint_matrix[:(2+model.nq*3),:].T,
                        constraint_vector[:(2+model.nq*3)])
                    ddq = u[:-1]
                    DDtrajectory_time=u[-1]
                    #print("QP infeasible – applying fallback damping.")
                    #exit(0)
                    #ddq = -10.0 * dq
                    #DDtrajectory_time = -10.0 * Dtrajectory_time
                else:
                    raise


            ddq = u[:-1]
            DDtrajectory_time = u[-1]

            elapsed = time.perf_counter() - t_qp_1
            ct_qp.append(elapsed)

            # --------------------------- INTEGRATION ----------------------------
            t += Tc

            q += dq * Tc + 0.5 * ddq * Tc ** 2
            dq += ddq * Tc

            trajectory_time += Dtrajectory_time * Tc + 0.5 * DDtrajectory_time * Tc ** 2.0
            Dtrajectory_time += DDtrajectory_time * Tc
            if USE_BRIDGE:
                bridge.sendCommand(q)

            # ----------------------------- TIMING -------------------------------
            elapsed = time.perf_counter() - loop_start
            ct.append(min(50e-3,elapsed))
            scaling_log.append(Dtrajectory_time)
            rest = Tc - elapsed
            if rest > 0:
                vizualization_string = (
                    f"h = {h_min:.2f} m, "
                    f"scaling {Dtrajectory_time:4.3f}, "
                    f"trajectory_error = {trajectory_error:.2f}"
                )

                #vizualization_string = f"h = {h_min:.2f} m, scaling {Dtrajectory_time:4.3f}, trajectory_error={trajectory_error:.2f}"
                renderer.push_state(q,
                                    Tbt_nominal,
                                    obstacle_positions,
                                    vizualization_string)
                elapsed = time.perf_counter() - loop_start
                rest = max(0.0,Tc - elapsed)

                time.sleep(rest)
            else:
                timeout_cycles+=1

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")


    # Call with your
    computation_times_planner = np.array(ct_planner)
    computation_times_qp = np.array(ct_qp)
    computation_times_ssm = np.array(ct_ssm)
    computation_times_pin = np.array(ct_pin)
    computation_times = np.array(ct)
    scaling_log = np.array(scaling_log)
    h_log = np.array(h_log)
    trj_error_log = np.array(trj_error_log)

    computation_times_others=computation_times-(computation_times_planner+computation_times_pin+computation_times_qp+computation_times_ssm)
    stats = {
        "computation_times": computation_times,
        "computation_times_qp": computation_times_qp,
        "computation_times_ssm": computation_times_ssm,
        "computation_times_planner": computation_times_planner,
        "computation_times_pin": computation_times_pin,
        "computation_times_others": computation_times_others,
    }

    print(f"timeout cycles = {timeout_cycles} over {cycles}, percentage = {100.0*timeout_cycles/cycles}, average = {np.mean(computation_times)}")
    print_stats_table(stats)
    _ = make_summary_figure(
        computation_times,
        computation_times_qp,
        computation_times_pin,
        computation_times_ssm,
        computation_times_others,
        h_log,
        trj_error_log,
        scaling_log,
    )


if __name__ == "__main__":
    main()
