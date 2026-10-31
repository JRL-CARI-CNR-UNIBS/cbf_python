"""Unit tests for robot command bridges."""

import numpy as np
import pytest

from cbf_python.bridges.fake_bridge import FakeCommandBridge, MultiRegionGestureSequencer
from cbf_python.bridges.joint_bridge import JointStateCommandBridge
from cbf_python.utils.simulation_helpers import HOME, UR10E_JOINTS


def test_fake_command_bridge():
    """Verify FakeCommandBridge positions and obstacle outputs."""
    bridge = FakeCommandBridge(ordered_joint_names=UR10E_JOINTS)
    bridge.sendCommand(HOME)

    q = bridge.getPositions()
    np.testing.assert_allclose(q, HOME, atol=1e-6)

    dq = bridge.getVelocities()
    assert dq.shape == (6,)

    pos, vel, acc = bridge.getObstacles(elapsed=0.0)
    assert isinstance(pos, np.ndarray)
    assert isinstance(vel, np.ndarray)
    assert isinstance(acc, np.ndarray)

    bridge.shutdown()


def test_bridge_threshold_validation():
    """Verify threshold exceeding raises ValueError."""
    bridge = FakeCommandBridge(ordered_joint_names=UR10E_JOINTS, threshold=0.1)
    bridge.sendCommand(HOME)

    # Jump of 1.0 rad exceeds threshold 0.1
    jump_q = HOME + 1.0
    with pytest.raises(ValueError, match="exceeds configured threshold"):
        bridge.sendCommand(jump_q)


def test_multi_region_gesture_sequencer():
    """Verify predictive multi-region gesture sequencing advances correctly."""
    bridge = FakeCommandBridge(ordered_joint_names=UR10E_JOINTS)
    sequencer = MultiRegionGestureSequencer(bridge, T_total=10.0)

    assert len(sequencer.gestures) == 3

    # Step at t=0, robot at s=0 -> should be waiting at rest
    pos, vel, acc, ev = sequencer.step(t=0.0, trajectory_time=0.0, curr_scaling=1.0)
    assert pos.shape[1] == 3
    assert np.all(vel == 0.0)
    assert ev is None

    # Step when robot approaches target 0 (target_s = 4.75, lead_time = 2.30)
    # At curr_s = 2.50, t_eta = 2.25 <= 2.30 -> should trigger!
    pos, vel, acc, ev = sequencer.step(t=2.50, trajectory_time=2.50, curr_scaling=1.0)
    assert ev is not None
    assert "Center Workstation Reach" in ev
    assert sequencer.active_gesture is not None

    bridge.shutdown()
