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
from visualization_daemon import VisualizationDaemon
from pinocchio import SE3
import math

# ---------------------------- CONSTANTS --------------------------------------
C = 0.25  # [m]  minimum separation distance
Tr = 0.15  # [s]  controller‑reaction time
a_s = 2.5  # [m/s²] robot decel/accel capability
v_h = 1.6  # [m/s]  assumed human approach speed
Tc = 2e-3  # [s]   2 kHz control period

gamma = 5.0  # CBF gain

# -------------------------- UTILITY FUNCTIONS --------------------------------

def compute_h(d, v, C=C, Tr=Tr, a_s=a_s, v_h = 0):
    """Inverse equation: minimum separation needed to permit speed |v|."""
    h=0.0
    if v < 0.0:
        if v_h>0:
            dmin = C + v_h * Tr - v * Tr + v_h * (-v / a_s) + 0.5 * v ** 2 / a_s
            h = d - dmin
        elif v_h<v:
            h = d - C
        else:
            h = d - C + (v - v_h)*Tr - (v_h-v)**2*0.5/a_s

    else:
        if v_h < 0:
            dmin = C
            coef = Tr
        else:
            dmin = C + v_h * Tr
            coef = Tr+v_h/a_s

        if d < dmin:
            h = coef*v
        else:
            #x = np.array([d-dmin, coef*v])
            #h = np.linalg.norm(x, ord=1)
            h = (d-dmin)+coef*v

    return h

def dh_dx(d, v, C=C, Tr=Tr, a_s=a_s, v_h = 0):
    """Derivative ∂h/∂v_rel used in CBF."""
    h = 0.0
    if v < 0.0:
        if v_h>0:
            derivative_h_on_distance = 1.0
            derivative_h_on_velocity = -(-Tr + (v - v_h) / a_s)  # Tr + (vh-v)/as = Tr+vh/as - v/as
        elif v_h<v:
            derivative_h_on_distance = 1.0
            derivative_h_on_velocity = 0.0
        else:
            derivative_h_on_distance = 1.0
            derivative_h_on_velocity = Tr + (v_h-v)/a_s


    else:
        if v_h < 0:
            dmin = C
            coef = Tr
        else:
            dmin = C + v_h * Tr
            coef = Tr + v_h / a_s
        if d < dmin:
            derivative_h_on_distance = 0.0
            derivative_h_on_velocity = coef
        else:
            derivative_h_on_distance = 1.0
            derivative_h_on_velocity = coef

    return derivative_h_on_distance, derivative_h_on_velocity



def range_state_derivative(r: np.ndarray, v: np.ndarray, eps: float = 1e-12):
    """Range‑differential dynamics: ḋ and d̈ as a linear function of a."""
    d = np.linalg.norm(r)
    if d < eps:
        raise ValueError("Range is zero; direction is undefined.")
    f1 = np.dot(r, v) / d
    f2 = np.dot(v, v) / d - np.dot(r, v) ** 2 / (d ** 3)
    g2 = r.T / d
    f = np.array([f1, f2]).reshape(-1, 1)
    g = np.vstack([np.zeros((1, 3)), g2])
    return f, g


def damped_pinv_svd(J, lam=1e-4):
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_damped = S / (S ** 2 + lam ** 2) # approssimazione di S^-1
    return (Vt.T * S_damped) @ U.T

def main():
    # --------------------------- MODEL & VISUALS ---------------------------------
    model_wrapper = load("ur10")
    model = model_wrapper.model
    viz = MeshcatVisualizer(model, model_wrapper.collision_model, model_wrapper.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Obstacles (red spheres)
    obstacle_positions: List[np.ndarray] = [np.array([0.8, 0.7, 0.5])]

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


    CBF = False
    renderer = VisualizationDaemon(viz)   # default 60 Hz



    # --------------------------- CONTROL INITIALISATION --------------------------
    data = model.createData()
    q = np.zeros(model.nq)
    q[1] = -np.pi / 2
    q[2] = np.pi / 2
    q[4] = np.pi / 4  # initial error

    dq = np.zeros(model.nq)

    ddq = np.zeros(model.nq)

    tool_frame_id = model.getFrameId("tool0")

    pin.framesForwardKinematics(model, data, q)

    # Gains
    wn=300
    xi=0.9
    Kp_tra = np.array([1, 1, 1])*wn**2
    Kd_tra = np.array([1,1,1])*2.0*xi*wn
    Kp_rot = np.array([1, 1, 1])*wn**2
    Kd_rot = np.array([1,1,1])*2.0*xi*wn

    twist_goal = np.zeros(6)


    planner = SegmentedSE3Trap(vlin_max=0.6, vang_max=1.2,
                                   alin_max=1.8, aang_max=2.0)


    def pose_eul(z, y, x, xyz):
        R = pin.utils.rotate('z', z) @ pin.utils.rotate('y', y) @ pin.utils.rotate('x', x)
        return SE3(R, np.array(xyz))

    goal_pose =  data.oMf[tool_frame_id].copy()
    # 2 · add way‑points -------------------------------------------
    planner.addWayPoint(goal_pose*SE3.Identity())  # start: no rotation, origin
    planner.addWayPoint(goal_pose*pose_eul(0.0, 0.0, 0.0, [0.30, 0.00, 0.0]))  # pure translation
    planner.addWayPoint(goal_pose*pose_eul(math.pi / 4, 0.0, 0.0, [0.30, -0.1, 0.020]))  # 45° about Z while moving
    planner.addWayPoint(goal_pose*pose_eul(math.pi / 4, 0.0, -math.pi / 4, [0.3, -0.1, 0.2]))  # add Y/X tilt
    planner.addWayPoint(goal_pose*pose_eul(-math.pi/4, 0.0, 0.0, [0.30, 0.0, 0.0]))  # spin 180° back + arc
    planner.addWayPoint(goal_pose*SE3.Identity())  # start: no rotation, origin
    T_total = planner.computeTime()

    renderer.publishPath(planner.publishPath())
    print(f"Total time = {T_total:.3f} s")
    # ------------------------------ MAIN LOOP -------------------- ----------------
    try:
        t = 0.0
        while t < 150.0:
            loop_start = time.perf_counter()

            goal_act_pose, twist_goal, goal_dtwist = planner.getMotionLaw(t % T_total)

            # CBF toggle (10 s on / 10 s off)
            CBF = (t % 40) < 20.0

            G=goal_act_pose.translation

            Rbg=goal_act_pose.rotation.copy()
            # Robot kinematics
            pin.framesForwardKinematics(model, data, q)
            pin.computeForwardKinematicsDerivatives(model, data, q, dq, ddq)

            Tbt = data.oMf[tool_frame_id]
            translation_bt = Tbt.translation
            Rbt = Tbt.rotation.copy()

            # Orientation error
            Rtg = Rbt.T @ Rbg
            error_rot = Rbt @ pin.log3(Rtg)

            # Current twist
            twist = pin.getFrameVelocity(
                model, data, tool_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            vel_lineare = twist.linear
            vel_angolare = twist.angular

            # Jacobians
            J = pin.computeFrameJacobian(
                model,
                data,
                q,
                tool_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            dJ = pin.frameJacobianTimeVariation(
                model,
                data,
                q,
                dq,
                tool_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            Jlin = J[:3, :]
            dJlin = dJ[:3, :]

            # Desired Cartesian accelerations
            acc_lin = Kp_tra * (G - translation_bt) + Kd_tra * (twist_goal[:3] - vel_lineare)
            acc_ang = Kp_rot * error_rot + Kd_rot * (twist_goal[3:] - vel_angolare)
            dtwist_tool = np.hstack([acc_lin, acc_ang])  #+goal_dtwist


            # ------------------------- CBF QP SET‑UP ------------------------------
            constraint_matrix = np.empty((0, model.nq))
            constraint_vector = np.empty((0, 1))

            # append constraints on joint position, joint velocity, joint torque

            for i, obs_pos in enumerate(obstacle_positions):
                # update obstacle motion
                w1=2 * np.pi/2
                w2=2 * np.pi/2.1
                obs_pos[0] = 0.8 - 0.25 * np.sin(w1* t)
                obs_pos[1] = 0.4 + 0.1 * np.sin(w2 * t)

                v_obs=np.array([0]*3)
                v_obs[0] = -0.25 * np.cos(w1* t)*w1
                v_obs[1] = 0.1 * np.cos(w2 * t)*w2



                r = translation_bt - obs_pos
                distance = np.linalg.norm(r)
                u_hr = r / distance

                v_h = np.dot(u_hr,v_obs)

                v_rel = np.dot(vel_lineare, u_hr)

                h=compute_h(d=distance, v=v_rel, v_h=v_h)

                f, g = range_state_derivative(r, vel_lineare)
                derivative_h_on_distance, derivative_h_on_velocity = dh_dx(d=distance, v=v_rel, v_h=v_h)

                partial_h_on_x = np.array([derivative_h_on_distance, derivative_h_on_velocity]).reshape(1, -1)
                Lie_f_h = partial_h_on_x @ f
                Lie_g_h = partial_h_on_x @ g

                constraint_matrix = np.append(constraint_matrix, Lie_g_h @ Jlin, axis=0)
                constraint_vector = np.append(
                    constraint_vector,
                    -Lie_g_h @ dJlin @ dq - Lie_f_h - gamma * h,
                    axis=0,
                )

            # ----------------------------- QP SOLVE -----------------------------
            P = J.T @ J
            b = (J.T @ (dtwist_tool - dJ @ dq)).flatten()
            constraint_vector = constraint_vector.flatten()

            if CBF:
                try:
                    ddq, *_ = quadprog.solve_qp(
                        P,
                        b,
                        constraint_matrix.T,
                        constraint_vector,
                        0,
                    )
                except ValueError as err:
                    if "constraints are inconsistent" in str(err):
                        print("QP infeasible – applying fallback damping.")
                        ddq = -10.0 * dq
                    else:
                        raise
            else:
                # no CBF, ddq calcolata in base a dtwist_tool
                # dtwist_tool (acc cart) = J @ ddq + dJ @ Dq
                ddq = damped_pinv_svd(J) @ (dtwist_tool - dJ @ dq)

            # --------------------------- INTEGRATION ----------------------------
            q += dq * Tc + 0.5 * ddq * Tc**2
            dq += ddq * Tc


            vizualization_string=f"h = {h:.2f} m, CBF={CBF}"
            renderer.push_state(q,
                                goal_act_pose,
                                obstacle_positions,
                                vizualization_string)
            # ----------------------------- TIMING -------------------------------
            t += Tc
            elapsed = time.perf_counter() - loop_start
            rest = Tc - elapsed
            if rest > 0:
                time.sleep(rest)

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")


if __name__ == "__main__":
    main()