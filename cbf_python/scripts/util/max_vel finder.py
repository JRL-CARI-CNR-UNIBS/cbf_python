#!/usr/bin/env python3
import numpy as np
import pinocchio as pin
from sharework import loadSharework


# ------------------------------------------------------------------------
# Joint-space limits (your definitions)
# ------------------------------------------------------------------------
def make_joint_limits(nv: int):
    """
    Create joint velocity and acceleration limits for nv joints, using
    your pattern:

        Dq_max  = pi * [1,1,...] * pi
        DDq_max = Dq_max * 5.0

    Returns
    -------
    Dq_max  : (nv,)
    DDq_max : (nv,)
    """
    base = np.ones(nv, dtype=np.float64)
    Dq_max = np.pi * base * np.pi          # = pi^2 for each joint
    DDq_max = Dq_max * 5.0                 # = 5 * pi^2 for each joint
    return Dq_max, DDq_max


# ------------------------------------------------------------------------
# Cartesian limits at a single configuration
# ------------------------------------------------------------------------
def cartesian_limits_at_q(model, data, q, frame_id, Dq_max, DDq_max):
    """
    Compute scalar Cartesian limits at configuration q:

      - max linear velocity   (‖v‖)
      - max angular velocity  (‖ω‖)
      - max linear accel      (‖a‖)
      - max angular accel     (‖α‖)

    induced by joint-space "limits" Dq_max and DDq_max.

    We interpret:
        dq   = Dq_max   (worst-case joint velocities)
        ddq  = DDq_max  (worst-case joint accelerations)

    Then:
        twist_max  = J(q) * dq
        dtwist_max = J(q) * ddq + dJ(q, dq) * dq
    """
    dq = Dq_max
    ddq = DDq_max

    # Forward kinematics & Jacobians
    pin.forwardKinematics(model, data, q, dq, ddq)
    pin.computeJointJacobians(model, data)

    J = pin.computeFrameJacobian(
        model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    dJ = pin.frameJacobianTimeVariation(
        model, data, q, dq, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    # Hypothetical max twist and accel if joints moved at their limits
    twist_max = J @ dq                       # [v; ω]
    dtwist_max = J @ ddq + dJ @ dq           # [a; α]

    v_lin = twist_max[:3]
    w_ang = twist_max[3:]
    a_lin = dtwist_max[:3]
    alpha = dtwist_max[3:]

    # Scalars = norms
    v_max_lin = float(np.linalg.norm(v_lin))
    w_max_ang = float(np.linalg.norm(w_ang))
    a_max_lin = float(np.linalg.norm(a_lin))
    alpha_max = float(np.linalg.norm(alpha))

    return v_max_lin, w_max_ang, a_max_lin, alpha_max


# ------------------------------------------------------------------------
# Global-ish Cartesian limits by random sampling
# ------------------------------------------------------------------------
def sample_cartesian_limits(model, frame_id, n_samples=1000, seed=0):
    """
    Randomly samples joint configurations within model joint bounds,
    and returns the maximum Cartesian norms induced by Dq_max and DDq_max.

    Returns
    -------
    v_max_lin_glob   : float   (m/s)
    w_max_ang_glob   : float   (rad/s)
    a_max_lin_glob   : float   (m/s^2)
    alpha_max_glob   : float   (rad/s^2)
    """
    np.random.seed(seed)
    data = model.createData()

    nv = model.nv
    Dq_max, DDq_max = make_joint_limits(nv)

    # Joint bounds (position)
    q_min = model.lowerPositionLimit
    q_max = model.upperPositionLimit

    v_max_lin_glob = 0.0
    w_max_ang_glob = 0.0
    a_max_lin_glob = 0.0
    alpha_max_glob = 0.0

    for _ in range(n_samples):
        # Random config in joint bounds
        q = q_min + (q_max - q_min) * np.random.rand(model.nq)

        v_lin, w_ang, a_lin, alpha = cartesian_limits_at_q(
            model, data, q, frame_id, Dq_max, DDq_max
        )

        if v_lin > v_max_lin_glob:
            v_max_lin_glob = v_lin
        if w_ang > w_max_ang_glob:
            w_max_ang_glob = w_ang
        if a_lin > a_max_lin_glob:
            a_max_lin_glob = a_lin
        if alpha > alpha_max_glob:
            alpha_max_glob = alpha

    return v_max_lin_glob, w_max_ang_glob, a_max_lin_glob, alpha_max_glob


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main():
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

    # Use the tool frame as Cartesian point of interest
    frame_name = "ur10e_wrist_3_joint"
    frame_id = model.getFrameId(frame_name)

    # Number of random samples
    n_samples = 200000

    print(f"Sampling Cartesian limits for {n_samples} random configurations...")
    v_lin, w_ang, a_lin, alpha = sample_cartesian_limits(
        model, frame_id, n_samples=n_samples, seed=0
    )

    print("\nApproximate global Cartesian limits induced by joint limits:")
    print(f"  Max linear velocity   v_max  ≈ {v_lin:.4f}  [m/s]")
    print(f"  Max angular velocity  ω_max  ≈ {w_ang:.4f}  [rad/s]")
    print(f"  Max linear accel      a_max  ≈ {a_lin:.4f}  [m/s^2]")
    print(f"  Max angular accel     α_max  ≈ {alpha:.4f}  [rad/s^2]")


if __name__ == "__main__":
    main()
