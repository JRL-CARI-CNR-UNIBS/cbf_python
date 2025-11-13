#!/usr/bin/env python3
# joint_target_publisher.py

import rclpy
from rclpy.node import Node
from rclpy.context import Context
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from builtin_interfaces.msg import Time
from contextlib import contextmanager
from typing import Sequence, Optional, overload
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import time
def _to_list(x):
    # Accept numpy arrays or anything sequence-like; fall back to single-item list
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        # This will turn numpy arrays etc. into lists
        return list(x)
    except Exception:
        return [x]

class JointTargetPublisher(Node):
    """
    Minimal publisher that exposes publish_once(q, dq, ddq).
    - q   -> JointState.position
    - dq  -> JointState.velocity
    - ddq -> JointState.effort
    """
    
    def __init__(
        self,
        topic: str = 'joint_target',
        joint_names: Optional[Sequence[str]] = None,
        frame_id: str = ''
    ):
        super().__init__('joint_target_publisher')
        self.pub = self.create_publisher(JointState, topic, 10)
        self.joint_names = list(joint_names) if joint_names is not None else None
        self.frame_id = frame_id
    
    def publish_once(self, q, dq, ddq):
        q  = _to_list(q)
        dq = _to_list(dq)
        ddq = _to_list(ddq)

        n = len(q)
        if len(dq) != n or len(ddq) != n:
            raise ValueError(
                f'Length mismatch: position={len(q)}, velocity={len(dq)}, effort={len(ddq)}'
            )
        if self.joint_names is not None and len(self.joint_names) != n:
            raise ValueError(
                f'joint_names length ({len(self.joint_names)}) must match data length ({n}).'
            )

        msg = JointState()
        now = self.get_clock().now().to_msg()  # builtin_interfaces/Time
        msg.header.stamp = Time(sec=now.sec, nanosec=now.nanosec)
        msg.header.frame_id = self.frame_id

        if self.joint_names is not None:
            msg.name = list(self.joint_names)

        msg.position = q
        msg.velocity = dq
        msg.effort   = ddq

        self.pub.publish(msg)

        
class TestStartPublisher(Node):
    """
    Minimal publisher that exposes publish_once(bool)
    """
    def __init__(
        self,
        topic: str = 'test_start'
    ):
        super().__init__('test_start_publisher')
        self.publisher = self.create_publisher(Bool, topic, 10)
        
    def publish_once(self, bool_value: bool):
    
        msg = Bool()
        msg.data = bool_value
        self.publisher.publish(msg)
        # Give the executor a chance to process the outgoing message if you’re exiting right away.
        #rclpy.spin_once(self, timeout_sec=0.0)

def publish_test_start_once(value: bool, topic: str = 'test_start', wait_match_sec: float = 1.0, wait_acked_sec: float = 0.5):
    """
    Publish a single Bool on `topic` from an isolated rclpy Context.
    - Waits up to `wait_match_sec` for at least one subscription to match.
    - After publish, waits up to `wait_acked_sec` for ACKs (if supported and RELIABLE).
    """
    # QoS: RELIABLE so we can wait_for_all_acked; KEEP_LAST(1) is enough.
    qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        durability=DurabilityPolicy.VOLATILE
    )

    ctx = Context()
    ctx.init()
    node = None
    exec_ = None
    try:
        node = Node('test_start_one_shot', context=ctx)
        pub = node.create_publisher(Bool, topic, qos)

        # private executor for this context
        exec_ = SingleThreadedExecutor(context=ctx)
        exec_.add_node(node)

        # 1) Wait for at least one subscription to match (discovery)
        t0 = time.time()
        while time.time() - t0 < wait_match_sec and pub.get_subscription_count() == 0:
            exec_.spin_once(timeout_sec=0.05)

        # 2) Publish
        msg = Bool()
        msg.data = value
        pub.publish(msg)

        # 3) Give DDS a breath + wait for ACKs (if available)
        try:
            # Only works if RELIABLE; ok to call even with 0 matched subs.
            pub.wait_for_all_acked(timeout_sec=wait_acked_sec)
        except Exception:
            # Fallback: a tiny spin/sleep to let the packet go out
            t1 = time.time()
            while time.time() - t1 < wait_acked_sec:
                exec_.spin_once(timeout_sec=0.05)

        # Optional: print for debug
        # node.get_logger().info(f"Published {topic}: {value} (subs={pub.get_subscription_count()})")

    finally:
        if exec_ is not None:
            try:
                exec_.shutdown()
            except Exception:
                pass
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        ctx.shutdown()

# You can run the module directly to test a single publish.
# if __name__ == '__main__':
#     # example data
#     q   = [0.0, 0.5, -0.3]
#     dq  = [0.0, 0.0,  0.0]
#     ddq = [0.0, -0.1, 0.1]
#     publisher = JointTargetPublisher(
#         topic='joint_target',
#         joint_names=['joint1', 'joint2', 'joint3'],
#         frame_id='base_link'
#     )
#     publisher.publish_once(q, dq, ddq)