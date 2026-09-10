"""Utilities for computing joint-space and induced Cartesian operational limits."""

from typing import Tuple
import numpy as np
import pinocchio as pin


def make_joint_limits(nv: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create joint velocity and acceleration limit vectors for nv joints.

    Limits are set to:
        Dq_max  = pi * [1, 1, ...] * pi = pi^2
        DDq_max = Dq_max * 5.0 = 5 * pi^2

    Args:
        nv: Number of degrees of freedom / velocities.

    Returns:
        Tuple of (Dq_max, DDq_max).
    """
    base = np.ones(nv, dtype=np.float64)
    Dq_max = np.pi * base * np.pi
    DDq_max = Dq_max * 5.0
    return Dq_max, DDq_max


def cartesian_limits_at_q(
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    frame_id: int,
    Dq_max: np.ndarray,
    DDq_max: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute scalar Cartesian velocity and acceleration upper bounds at configuration q.

    Args:
        model: Pinocchio robot model.
        data: Pinocchio model data.
        q: Joint configuration vector.
        frame_id: Target frame index.
        Dq_max: Joint velocity limit vector.
        DDq_max: Joint acceleration limit vector.

    Returns:
        Tuple of (v_max_lin, w_max_ang, a_max_lin, alpha_max_ang).
    """
    dq = Dq_max
    ddq = DDq_max

    pin.forwardKinematics(model, data, q, dq, ddq)
    pin.computeJointJacobians(model, data)

    J = pin.computeFrameJacobian(
        model, data, q, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    dJ = pin.frameJacobianTimeVariation(
        model, data, q, dq, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    twist_max = J @ dq
    dtwist_max = J @ ddq + dJ @ dq

    v_lin = twist_max[:3]
    w_ang = twist_max[3:]
    a_lin = dtwist_max[:3]
    alpha = dtwist_max[3:]

    return (
        float(np.linalg.norm(v_lin)),
        float(np.linalg.norm(w_ang)),
        float(np.linalg.norm(a_lin)),
        float(np.linalg.norm(alpha)),
    )


def sample_cartesian_limits(
    model: pin.Model,
    frame_id: int,
    n_samples: int = 1000,
    seed: int = 0,
) -> Tuple[float, float, float, float]:
    """Sample random joint configurations within bounds and find maximum Cartesian norms.

    Args:
        model: Pinocchio robot model.
        frame_id: Target frame index.
        n_samples: Number of uniform random joint samples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (v_max_lin_glob, w_max_ang_glob, a_max_lin_glob, alpha_max_glob).
    """
    np.random.seed(seed)
    data = model.createData()

    nv = model.nv
    Dq_max, DDq_max = make_joint_limits(nv)

    q_min = model.lowerPositionLimit
    q_max = model.upperPositionLimit

    v_max_lin_glob = 0.0
    w_max_ang_glob = 0.0
    a_max_lin_glob = 0.0
    alpha_max_glob = 0.0

    for _ in range(n_samples):
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


def main() -> None:
    from sharework import loadSharework

    UR10E_JOINTS = [
        "ur10e_shoulder_pan_joint",
        "ur10e_shoulder_lift_joint",
        "ur10e_elbow_joint",
        "ur10e_wrist_1_joint",
        "ur10e_wrist_2_joint",
        "ur10e_wrist_3_joint",
    ]

    model_wrapper = loadSharework(UR10E_JOINTS)
    model = model_wrapper.model
    frame_name = "ur10e_wrist_3_joint"
    frame_id = model.getFrameId(frame_name)

    n_samples = 2000
    print(f"Sampling Cartesian limits for {n_samples} configurations...")
    v_lin, w_ang, a_lin, alpha = sample_cartesian_limits(model, frame_id, n_samples=n_samples)

    print("\nCartesian limits induced by joint limits:")
    print(f"  Max linear velocity   v_max  ≈ {v_lin:.4f} [m/s]")
    print(f"  Max angular velocity  w_max  ≈ {w_ang:.4f} [rad/s]")
    print(f"  Max linear accel      a_max  ≈ {a_lin:.4f} [m/s^2]")
    print(f"  Max angular accel     alpha  ≈ {alpha:.4f} [rad/s^2]")


if __name__ == "__main__":
    main()
