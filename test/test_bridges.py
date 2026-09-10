"""Unit tests for robot command bridges."""

import numpy as np
import pytest

from cbf_python.bridges.fake_bridge import FakeCommandBridge
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
