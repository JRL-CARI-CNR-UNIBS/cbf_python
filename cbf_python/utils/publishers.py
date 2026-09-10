"""
Asynchronous publishers and CSV loggers for robot joint targets and diagnostics.
"""

from __future__ import annotations
import csv
import queue
import threading
import time
from pathlib import Path
from typing import Sequence, Optional, Union, List, Any

import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool
from builtin_interfaces.msg import Time as MsgTime


def str_to_list(val: Union[str, Sequence[str]]) -> List[str]:
    """Convert comma-separated string or list to list of clean strings."""
    if isinstance(val, str):
        return [s.strip() for s in val.split(",") if s.strip()]
    return [str(s).strip() for s in val]


class _AsyncPublishBus:
    """Thread-safe background execution bus for asynchronous publishing and I/O."""

    def __init__(self, maxsize: int = 4000, name: str = "AsyncPublishBusWorker") -> None:
        self._queue: queue.Queue[Optional[tuple]] = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=name,
            daemon=True,
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self._queue.task_done()
                break

            fn, args, kwargs = item
            try:
                fn(*args, **kwargs)
            except Exception as e:
                pass
            finally:
                self._queue.task_done()

    def submit(
        self,
        fn: Any,
        *args: Any,
        block: bool = False,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> bool:
        if self._stop_event.is_set():
            return False
        try:
            self._queue.put((fn, args, kwargs), block=block, timeout=timeout)
            return True
        except queue.Full:
            return False

    def shutdown(self, wait: bool = True) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if wait and self._worker.is_alive():
            self._worker.join(timeout=2.0)


_GLOBAL_BUS: Optional[_AsyncPublishBus] = None
_BUS_LOCK = threading.Lock()


def get_global_bus() -> _AsyncPublishBus:
    global _GLOBAL_BUS
    with _BUS_LOCK:
        if _GLOBAL_BUS is None:
            _GLOBAL_BUS = _AsyncPublishBus()
        return _GLOBAL_BUS


class _CsvWriter:
    """Thread-safe CSV row appender."""

    def __init__(self, csv_path: Union[str, Path], column_names: Union[str, Sequence[str]]) -> None:
        self.csv_path = Path(csv_path)
        self.column_names = str_to_list(column_names)
        self._lock = threading.Lock()
        self._ensure_header_written()

    def _ensure_header_written(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self._lock:
                with self.csv_path.open(mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.column_names)

    def append_row(self, t: float, row: Sequence[Any]) -> None:
        final_row = [str(t)] + [str(x) for x in row]
        with self._lock:
            with self.csv_path.open(mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(final_row)


class JointTargetCsvPublisher:
    """Asynchronous CSV logger for commanded joint trajectory states."""

    def __init__(
        self,
        csv_path: Union[str, Path],
        column_names: Union[str, Sequence[str]],
        joint_names: Sequence[str],
        bus: Optional[_AsyncPublishBus] = None,
    ) -> None:
        self._writer = _CsvWriter(csv_path, column_names)
        self.joint_names = list(joint_names)
        self._bus = bus or get_global_bus()

    def publish_once(
        self,
        t: float,
        q: Sequence[float],
        dq: Optional[Sequence[float]] = None,
        ddq: Optional[Sequence[float]] = None,
    ) -> None:
        q_list = list(q)
        dq_list = list(dq) if dq is not None else [0.0] * len(q_list)
        ddq_list = list(ddq) if ddq is not None else [0.0] * len(q_list)
        row = []
        for i in range(len(self.joint_names)):
            row.extend([q_list[i], dq_list[i], ddq_list[i]])
        self._bus.submit(self._writer.append_row, t, row)


class DoubleArrayCsvPublisher:
    """Asynchronous CSV logger for numeric data arrays."""

    def __init__(
        self,
        csv_path: Union[str, Path],
        column_names: Union[str, Sequence[str]],
        bus: Optional[_AsyncPublishBus] = None,
    ) -> None:
        self._writer = _CsvWriter(csv_path, column_names)
        self._bus = bus or get_global_bus()

    def publish_once(self, t: float, data: Sequence[float]) -> None:
        self._bus.submit(self._writer.append_row, t, list(data))


class TestStartCsvPublisher:
    """Asynchronous CSV logger for test start markers."""

    def __init__(
        self,
        csv_path: Union[str, Path],
        column_names: Union[str, Sequence[str]] = ("time", "val"),
        bus: Optional[_AsyncPublishBus] = None,
    ) -> None:
        self._writer = _CsvWriter(csv_path, column_names)
        self._bus = bus or get_global_bus()

    def publish_once(self, val: bool = True, t: Optional[float] = None) -> None:
        t_val = time.time() if t is None else float(t)
        self._bus.submit(self._writer.append_row, t_val, [bool(val)])


class JointTargetPublisher(Node):
    """ROS 2 Node for publishing joint targets over /joint_states or custom topics."""

    def __init__(
        self,
        topic: str = "joint_target",
        joint_names: Sequence[str] = (),
        frame_id: str = "world",
        node_name: str = "joint_target_publisher",
    ) -> None:
        super().__init__(node_name)
        self.joint_names = list(joint_names)
        self.frame_id = frame_id
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pub = self.create_publisher(JointState, topic, qos)
        self._bus = get_global_bus()

    def _publish_now(self, t: float, q: Sequence[float], dq: Sequence[float], ddq: Sequence[float]) -> None:
        msg = JointState()
        sec = int(t)
        nanosec = int((t - sec) * 1e9)
        msg.header.stamp = MsgTime(sec=sec, nanosec=nanosec)
        msg.header.frame_id = self.frame_id
        msg.name = self.joint_names
        msg.position = [float(x) for x in q]
        msg.velocity = [float(x) for x in dq] if dq is not None else []
        msg.effort = [float(x) for x in ddq] if ddq is not None else []
        self._pub.publish(msg)

    def publish_once(
        self,
        t: float,
        q: Sequence[float],
        dq: Optional[Sequence[float]] = None,
        ddq: Optional[Sequence[float]] = None,
    ) -> None:
        self._bus.submit(self._publish_now, t, q, dq, ddq)


class DoubleArrayPublisher(Node):
    """ROS 2 Node for publishing double array telemetry."""

    def __init__(
        self,
        topic: str = "cbf_output",
        node_name: str = "double_array_publisher",
    ) -> None:
        super().__init__(node_name)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pub = self.create_publisher(Float64MultiArray, topic, qos)
        self._bus = get_global_bus()

    def _publish_now(self, data: Sequence[float]) -> None:
        msg = Float64MultiArray()
        msg.data = [float(x) for x in data]
        self._pub.publish(msg)

    def publish_once(self, t: float, data: Sequence[float]) -> None:
        self._bus.submit(self._publish_now, data)


class TestStartPublisher(Node):
    """ROS 2 Node for publishing test start synchronization booleans."""

    def __init__(
        self,
        topic: str = "test_start",
        node_name: str = "test_start_publisher",
    ) -> None:
        super().__init__(node_name)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub = self.create_publisher(Bool, topic, qos)
        self._bus = get_global_bus()

    def _publish_now(self, val: bool) -> None:
        msg = Bool()
        msg.data = bool(val)
        self._pub.publish(msg)

    def publish_once(self, val: bool = True) -> None:
        self._bus.submit(self._publish_now, val)


def swap_csv(path_in: str, path_out: str, index_0: int, index_1: int) -> None:
    """Utility to swap two keypoint columns in a dataset CSV file."""
    df = pd.read_csv(path_in, header=0, index_col=False)
    suff_to_swap = ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc"]
    for suff in suff_to_swap:
        c0 = f"keypoint{index_0}_{suff}"
        c1 = f"keypoint{index_1}_{suff}"
        if c0 in df.columns and c1 in df.columns:
            df[c0], df[c1] = df[c1].copy(), df[c0].copy()
    df.to_csv(path_out, header=True, index=False)
