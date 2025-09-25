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

from interpolator import SegmentedSE3Trap
from joint_interpolator import SegmentedJointTrap
from visualization_daemon import VisualizationDaemon
from pinocchio import SE3
from ssm_cbf import *
from sharework import loadSharework

from human_pose_reader import  PoseReader

import math

# ---------------------------- CONSTANTS --------------------------------------
C = 0.25  # [m]  minimum separation distance
Tr = 0.15  # [s]  controller‑reaction time
a_s = 0.5  # [m/s²] robot decel/accel capability
Tc = 10e-3  # [s]   2 kHz control period

gamma = 50.0  # CBF gain
from scipy.linalg import block_diag


def damped_pinv_svd(J, lam=1e-4):
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_damped = S / (S ** 2 + lam ** 2)  # approssimazione di S^-1
    return (Vt.T * S_damped) @ U.T


def main():
    # --------------------------- MODEL & VISUALS ---------------------------------
    SHAREWORK = True
    if SHAREWORK:
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
    else:
        model_wrapper = load('ur10')
        prefix = ''
    model = model_wrapper.model
    viz = MeshcatVisualizer(model, model_wrapper.collision_model, model_wrapper.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    R = pin.utils.rotate('z', 1.9) @ pin.utils.rotate('x', 1.57)
    T_wc = pin.SE3(R, np.array([-1.45, -0.9, 0.9]))
    reader = PoseReader("a01_s10_e02_skeleton3D_converted.csv", T_wc)
    obstacle_positions = reader.getHumanPose(0)
    last_obstacle_positions = obstacle_positions.copy()
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
    q = np.zeros(model.nq)
    q[0] = np.pi / 2
    q[1] = -np.pi / 2
    q[2] = np.pi / 2
    q[4] = np.pi / 4  # initial error

    q2 = q.copy()
    q2[4] = np.pi
    q2[1] = 0
    q2[2] = -np.pi/2

    dq = np.zeros(model.nq)

    ddq = np.zeros(model.nq)

    tool_frame_id = model.getFrameId(prefix+"tool0")
    elbow_frame_id = model.getFrameId(prefix+"forearm_link")
    frames_ids=[elbow_frame_id,tool_frame_id]
    #frames_ids=[tool_frame_id]

    pin.framesForwardKinematics(model, data, q)

    # Gains
    wn = 300
    xi = 0.9
    Kp_tra = np.array([1, 1, 1]) * wn ** 2
    Kd_tra = np.array([1, 1, 1]) * 2.0 * xi * wn
    Kp_rot = np.array([1, 1, 1]) * wn ** 2
    Kd_rot = np.array([1, 1, 1]) * 2.0 * xi * wn

    twist_goal = np.zeros(6)
    scaling_limit_matrix = np.append(np.zeros(model.nq), Tc)

    Dq_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])*np.pi
    DDq_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])*np.pi*10

    planner = SegmentedJointTrap(Dq_max=Dq_max*.8, DDq_max=DDq_max*.8)

    def pose_eul(z, y, x, xyz):
        R = pin.utils.rotate('z', z) @ pin.utils.rotate('y', y) @ pin.utils.rotate('x', x)
        return SE3(R, np.array(xyz))

    goal_pose = data.oMf[tool_frame_id].copy()
    # 2 · add way‑points -------------------------------------------
    planner.addWayPoint(q)
    planner.addWayPoint(q2)
    planner.addWayPoint(q)

    T_total = planner.computeTime()

    renderer.publishPath(planner.publishPath())
    print(f"Total time = {T_total:.3f} s")

    I=np.eye(model.nq)

    timeout_cycles=0
    cycles=0
    ct, ct_qp, ct_ssm, ct_planner, ct_pin = [], [], [], [], []

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

    lambda1 = 1.0e2
    lambda2 = 1
    lambda3 = 1e-0
    lambda4 = 1e-9

    v_obs = np.array([0] * 3)
    zeros_nq = np.zeros(model.nq)
    zeros3 = np.zeros(3)
    b1 = np.zeros(model.nq + 1)
    b2 = np.zeros(model.nq + 1)
    b3 = np.zeros(model.nq + 1)
    b3[:model.nq] = 0.0  # fixed each loop except last entry
    P = np.empty((model.nq + 1, model.nq + 1))
    b = np.empty(model.nq + 1)

    J = np.zeros((6, model.nv))
    dJ = np.zeros((6, model.nv))

    # ------------------------------ MAIN LOOP -------------------- ----------------
    try:

        t = 0.0

        trajectory_time = 0.0
        Dtrajectory_time = 1.0
        DDtrajectory_time = 0.0

        n_constraints = 2 + len(obstacle_positions)*len(frames_ids)
        constraint_matrix = np.zeros((n_constraints, model.nq + 1))
        constraint_vector = np.zeros(n_constraints)
        constraint_matrix[0, :] = -scaling_limit_matrix
        constraint_matrix[1, :] = scaling_limit_matrix

        J = np.zeros((6, model.nv))
        dJ = np.zeros((6, model.nv))

        h_min = np.inf

        Lie_f_h = 0.0
        Lie_g_h = np.zeros(3)




        while t < 30.0:
            h_min = np.inf

            loop_start = time.perf_counter()
            last_obstacle_positions = obstacle_positions
            obstacle_positions = reader.getHumanPose(0.1*t)
            cycles += 1

            nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time % T_total)

            pin.framesForwardKinematics(model, data, nominal_q)
            Tbt_nominal = data.oMf[tool_frame_id]

            elapsed = time.perf_counter() - loop_start
            ct_planner.append(elapsed)

            # ------------------------- CBF QP SETUP -------------------------
            row_idx = 0  # reset index at each loop

            # Scaling constraints
            constraint_vector[row_idx] = -(1 - Dtrajectory_time)
            row_idx += 1

            constraint_vector[row_idx] = -Dtrajectory_time
            row_idx += 1

            t_pin_1 = time.perf_counter()

            pin.computeForwardKinematicsDerivatives(model, data, q, dq, ddq)

            elapsed_pin = time.perf_counter() - t_pin_1

            elapsed_ssm=0.0

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
                for i, (obs_pos, last_obs_pos) in enumerate(zip(obstacle_positions, last_obstacle_positions)):

                    # update obstacle motion

                    #v_obs =0.0*(obs_pos-last_obs_pos)/Tc
                    r = translation_bt - obs_pos
                    u_hr = r / np.linalg.norm(r)
                    v_h = np.dot(u_hr, v_obs)
                    v_rel = np.dot(u_hr, vel_lineare)

                    h = compute_h(d=np.linalg.norm(r), v=v_rel, v_h=v_h, C=C, Tr=Tr, a_s=a_s)
                    h_min = min(h_min,h)

                    #f, g = range_state_derivative(v_lin=vel_lineare, v_human=v_obs)
                    #Jh_psi = jacobian_h(d=distance, v=v_rel, v_h=v_h, C=C, Tr=Tr, a_s=a_s)
                    #Jpsi_chi = jacobian_psi(translation_bt, obs_pos, vel_lineare, v_obs)
                    #Jh_chi = Jh_psi @ Jpsi_chi

                    # partial_h_on_x = np.array([derivative_h_on_distance, derivative_h_on_velocity]).reshape(1, -1)
                    #Lie_f_h = Jh_chi @ f
                    #Lie_g_h = Jh_chi @ g

                    lie_fg_h_fast(
                        p_r=translation_bt, p_h=obs_pos,
                        v_lin=vel_lineare, v_human=v_obs,
                        C=C, Tr=Tr, a_s=a_s,
                        Lie_f_h_out=Lie_f_h, Lie_g_h_out=Lie_g_h
                    )

                    # Fill preallocated row
                    constraint_matrix[row_idx, :-1] = Lie_g_h @ Jlin
                    #constraint_matrix[row_idx, -1] = 0.0
                    constraint_vector[row_idx] = (-Lie_g_h @ dJlin @ dq - Lie_f_h - gamma * h)
                    row_idx += 1
                elapsed_ssm += time.perf_counter() - t_ssm_1
            ct_ssm.append(elapsed_ssm)
            ct_pin.append(elapsed_pin)

            # ----------------------------- QP SOLVE -----------------------------
            b1[:-1]=(nominal_q-q-dq*Tc)*0.5*Tc**2
            b3[-1] = -Tc*(Dtrajectory_time-1)

            P2[:model.nq, -1] = -(Tc ** 2) * nominal_Dq
            P2[-1, :model.nq] = P2[:model.nq, -1]
            P2[-1, -1] = Tc ** 2 * nominal_Dq.dot(nominal_Dq)

            b2[:-1] = (nominal_Dq*Dtrajectory_time-dq)*Tc
            b2[-1] = (nominal_Dq*Dtrajectory_time-dq).dot(nominal_Dq*Tc)

            P=lambda1*P1+lambda2*P2+lambda3*P3+lambda4*P4
            b=lambda1*b1+lambda2*b2+lambda3*b3+lambda4*b4
            try:
                t_qp_1 = time.perf_counter()
                u, *_ = quadprog.solve_qp(
                    P,
                    b,
                    constraint_matrix.T,
                    constraint_vector,
                    0,
                )
                elapsed = time.perf_counter() - t_qp_1
                ct_qp.append(elapsed)
                ddq = u[:-1]
                DDtrajectory_time=u[-1]
            except ValueError as err:
                if "constraints are inconsistent" in str(err):
                    print("QP infeasible – applying fallback damping.")
                    ddq = -10.0 * dq
                    DDtrajectory_time = -10.0 * Dtrajectory_time
                else:
                    raise

            # --------------------------- INTEGRATION ----------------------------
            t += Tc

            q += dq * Tc + 0.5 * ddq * Tc ** 2
            dq += ddq * Tc

            trajectory_time += Dtrajectory_time * Tc + 0.5 * DDtrajectory_time * Tc ** 2.0
            Dtrajectory_time += DDtrajectory_time * Tc


            # ----------------------------- TIMING -------------------------------
            elapsed = time.perf_counter() - loop_start
            ct.append(elapsed)
            rest = Tc - elapsed
            if 1: #rest > 0:
                vizualization_string = f"h = {h_min:.2f} m, scaling {Dtrajectory_time:4.3f}"
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


    def print_stats_table(stats):
        # Print header
        print(f"{'Name':<30} {'Mean':>12} {'50%':>12} {'90%':>12} {'95%':>12} {'99%':>12}")
        print("-" * 90)
        # Print each row
        for name, data in stats.items():
            mean_val = np.mean(data*1000)
            q50, q90, q95, q99 = np.quantile(data*1000, [0.50, 0.90, 0.95, 0.99])
            print(f"{name:<30} {mean_val:12.6f} {q50:12.6f} {q90:12.6f} {q95:12.6f} {q99:12.6f}")

    # Call with your
    computation_times_planner = np.array(ct_planner)
    computation_times_qp = np.array(ct_qp)
    computation_times_ssm = np.array(ct_ssm)
    computation_times_pin = np.array(ct_pin)
    computation_times = np.array(ct)
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

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))  # optional: makes the plot larger

    plt.hist(computation_times, bins=100, alpha=0.5, label="Computation Times")
    plt.hist(computation_times_qp, bins=100, alpha=0.5, label="Computation Times QP")
    plt.hist(computation_times_pin, bins=100, alpha=0.5, label="Computation Times PIN")
    plt.hist(computation_times_ssm, bins=100, alpha=0.5, label="Computation Times SSM")
    plt.hist(computation_times_others, bins=100, alpha=0.5, label="Computation Times Others")

    plt.xlabel("Computation Time")
    plt.ylabel("Frequency")
    plt.title("Comparison of Computation Times")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()