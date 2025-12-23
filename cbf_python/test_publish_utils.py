#!/usr/bin/env python3
# joint_target_publisher.py

import rclpy
from rclpy.node import Node
from rclpy.context import Context
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
from builtin_interfaces.msg import Time
from contextlib import contextmanager
from typing import Sequence, Optional, overload
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import time
import pandas as pd
import threading
import queue
import numpy as np
from typing import Optional, Sequence

# ... your existing imports ...
# from rclpy.node import Node
# from std_msgs.msg import Bool
# from sensor_msgs.msg import JointState
# etc.

# -------------------------------------------------------------------
# Global async publish bus: single queue, single worker thread
# -------------------------------------------------------------------

def ndarray2list(array):
    list_out = []
    for element in array:
        list_out.extend(element)
    return list_out

class _AsyncPublishBus:
    """
    Global bus that executes arbitrary callables on a single background thread.
    Items in the queue are (callable, args, kwargs).
    """
    def __init__(self, maxsize: int = 2000):
        self._queue: "queue.Queue[tuple]" = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="AsyncPublishBusWorker",
            daemon=True,
        )
        self._worker.start()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                fn, args, kwargs = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if fn is None:
                self._queue.task_done()
                break

            try:
                fn(*args, **kwargs)
            except Exception as e:
                # You can replace this with any logging you like
                print(f"[AsyncPublishBus] error in task {fn}: {e}")
            finally:
                self._queue.task_done()

    def submit(
        self,
        fn,
        *args,
        block: bool = False,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        """
        Enqueue a callable to be executed on the worker thread.
        If block=False, drop tasks when the queue is full.
        """
        if self._stop_event.is_set():
            return

        item = (fn, args, kwargs)
        if block:
            self._queue.put(item, timeout=timeout)
        else:
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                # Drop instead of blocking the caller
                pass

    def shutdown(self, wait: bool = True):
        self._stop_event.set()
        try:
            self._queue.put_nowait((None, (), {}))
        except queue.Full:
            pass
        if wait:
            try:
                self._worker.join(timeout=1.0)
            except Exception:
                pass


# Singleton instance getter
_bus_lock = threading.Lock()
_global_bus: Optional[_AsyncPublishBus] = None


def _get_global_bus() -> _AsyncPublishBus:
    global _global_bus
    with _bus_lock:
        if _global_bus is None:
            _global_bus = _AsyncPublishBus(maxsize=2000)
    return _global_bus


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

        # shared bus for all publishers:
        self._bus = _get_global_bus()

    
    def _publish_now(self, q, dq, ddq):
        q  = _to_list(q)
        dq = _to_list(dq)
        ddq = _to_list(ddq)

        n = len(q)
        # print(f"Publishing JointTarget of length: {n}")
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
    def publish_once(self, q, dq, ddq, *, block: bool = False, timeout: Optional[float] = None):
        """
        Public API (same signature as before, with extra optional args):
        - Non-blocking by default; drops messages if the queue is full.
        - Set block=True if you ever want back-pressure in non-RT contexts.
        """
        self._bus.submit(self._publish_now, q, dq, ddq, block=block, timeout=timeout)

class DoubleArrayPublisher(Node):
    """
    Minimal publisher that exposes publish_once(arr).
    """
    
    def __init__(
        self,
        topic: str = 'human_state',
        node_name: str = 'double_array_publisher',
        #dim : int = 0,
        frame_id: str = ''
    ):
        super().__init__(node_name)
        self.pub = self.create_publisher(Float64MultiArray, topic, 10)
        self.frame_id = frame_id
        #self.dim = dim
        # shared bus for all publishers:
        self._bus = _get_global_bus()

    
    def _publish_now(self, array):

        if isinstance(array, np.ndarray):
            array = ndarray2list(array)
            # print(array)
            # print("Length: ",len(array))
        else:
            array = _to_list(array)

        # n = len(array)
        # if n != self.dim:
        #     raise ValueError(
        #         f'Length mismatch: requested: {self.dim}, got: {len(array)}'
        #     )

        msg = Float64MultiArray()
        now = self.get_clock().now().nanoseconds/1e09  # builtin_interfaces/Time
        # self._logger.warning(f"Publishing at time: {now}, OF TYPE: {type(now)}")

        msg.data = []
        msg.data.append(now)
        msg.data.extend(array)
        # self._logger.warning(f"Publishing time: {msg.data[0]}")    
        self.pub.publish(msg)


    def publish_once(self, array, *, block: bool = False, timeout: Optional[float] = None):
        """
        Public API (same signature as before, with extra optional args):
        - Non-blocking by default; drops messages if the queue is full.
        - Set block=True if you ever want back-pressure in non-RT contexts.
        """
        self._bus.submit(self._publish_now, array, block=block, timeout=timeout)


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
        self._bus = _get_global_bus()

    def _publish_now(self, bool_value: bool):
    
        msg = Bool()
        msg.data = bool_value
        self.publisher.publish(msg)
        # Give the executor a chance to process the outgoing message if you’re exiting right away.
        #rclpy.spin_once(self, timeout_sec=0.0)

    def publish_once(self, bool_value: bool, *, block: bool = False, timeout: Optional[float] = None):
        """
        Enqueue a single Bool publish on the global bus.
        """
        self._bus.submit(self._publish_now, bool_value, block=block, timeout=timeout)

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