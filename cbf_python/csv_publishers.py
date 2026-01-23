#!/usr/bin/env python3
# async_csv_publishers.py

import csv
import threading
import queue
from pathlib import Path
from typing import Sequence, Optional, Any
from time import time
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Helper conversion functions
# -------------------------------------------------------------------
def str_to_list(string):
    return string.split(",")

def ndarray2list(array):
    """
    Flatten a numpy array of shape (N,) or (N, M) into a simple Python list.
    """
    list_out = []
    for element in array:
        # Try to extend if it's iterable (e.g. sub-array), else append
        try:
            list_out.extend(element)
        except TypeError:
            list_out.append(element)
    return list_out


def _to_list(x):
    """
    Accept numpy arrays or anything sequence-like; fall back to single-item list.
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        return list(x)
    except Exception:
        return [x]


# -------------------------------------------------------------------
# Thread-safe CSV writer
# -------------------------------------------------------------------

class _CsvWriter:
    """
    Simple CSV writer that:
    - Ensures the header is written once.
    - Appends rows in a thread-safe way.
    """

    def __init__(self, csv_path: str, column_names: str):
        self.csv_path = Path(csv_path)
        self.column_names = str_to_list(column_names)
        self._lock = threading.Lock()
        self._ensure_header_written()

    def _ensure_header_written(self):
        """
        If the file does not exist or is empty, write the header row.
        """
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self._lock:
                with self.csv_path.open(mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.column_names)

    def append_row(self,t, row: Sequence[Any]):
        """
        Append a single row (list/tuple) to the CSV file.
        """
        now = t
        final_row = []
        final_row.append(str(now))
        final_row.extend(row)
        if len(final_row) != len(self.column_names):
            raise ValueError(
                f"Row length {len(row)} does not match number of columns "
                f"{len(self.column_names)}"
                f"{final_row}"
            )
        with self._lock:
            with self.csv_path.open(mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(final_row)


# -------------------------------------------------------------------
# Global async publish bus: single queue, single worker thread
# -------------------------------------------------------------------

class _AsyncPublishBus:
    """
    Global bus that executes arbitrary callables on a single background thread.
    Items in the queue are (callable, args, kwargs).

    This mirrors the pattern you used in your ROS2 code.
    """

    def __init__(self, maxsize: int = 2000):
        self._queue: "queue.Queue[tuple]" = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="AsyncCsvPublishBusWorker",
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
                # Shutdown sentinel
                self._queue.task_done()
                break

            try:
                fn(*args, **kwargs)
            except Exception as e:
                # Replace with proper logging if you want
                print(f"[AsyncCsvPublishBus] error in task {fn}: {e}")
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

        Parameters
        ----------
        fn : callable
            Function to call in the worker thread.
        *args, **kwargs :
            Arguments passed to fn(*args, **kwargs).
        block : bool
            If False (default), drop tasks when the queue is full.
            If True, block until a slot is free or `timeout` expires.
        timeout : Optional[float]
            Timeout passed to queue.put() when block=True.
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
        """
        Stop the worker thread. Call this at program shutdown if you want
        a clean exit.
        """
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


# -------------------------------------------------------------------
# Publishers: CSV-based, using the global async bus
# -------------------------------------------------------------------

class JointTargetCsvPublisher:
    """
    CSV-based analogue of your ROS2 JointTargetPublisher.

    Instead of publishing a sensor_msgs/JointState, this writes one row to CSV:

        [q..., dq..., ddq...]

    The structure is determined by `column_names`, e.g.:

        ["q1", "q2", "q3", "dq1", "dq2", "dq3", "ddq1", "ddq2", "ddq3"]
    """

    def __init__(
        self,
        csv_path: str,
        column_names: str,
        joint_names: Optional[Sequence[str]] = None,
    ):
        """
        Parameters
        ----------
        csv_path : str
            Path to the CSV file to write to.
        column_names : Sequence[str]
            List of column names written as the CSV header.
            Must match the row length you generate (q + dq + ddq).
        joint_names : Optional[Sequence[str]]
            Optional list of joint names; only used for consistency checks.
        """
        self.joint_names = list(joint_names) if joint_names is not None else None
        self._writer = _CsvWriter(csv_path, column_names)
        self._bus = _get_global_bus()

    def _publish_now(self,t,  q, dq, ddq):
        """
        Synchronous implementation: immediately appends a row to the CSV.
        This is run in the background worker thread via the global bus.
        """
        q_list = _to_list(q)
        dq_list = _to_list(dq)
        ddq_list = _to_list(ddq)

        n = len(q_list)
        if len(dq_list) != n or len(ddq_list) != n:
            raise ValueError(
                f"Length mismatch: position={len(q_list)}, "
                f"velocity={len(dq_list)}, effort={len(ddq_list)}"
            )

        if self.joint_names is not None and len(self.joint_names) != n:
            raise ValueError(
                f"joint_names length ({len(self.joint_names)}) must match "
                f"data length ({n})."
            )

        row =[]
        for i in range(len(q_list)):
            row.append(q_list[i])
            row.append(dq_list[i])
            row.append(ddq_list[i])
        self._writer.append_row(t, row)

    def publish_once(
        self,
        t,
        q,
        dq,
        ddq,
        *,
        block: bool = False,
        timeout: Optional[float] = None,
    ):
        """
        Public API (async):

        - Enqueues a single row [q..., dq..., ddq...] to be written to CSV.
        - Non-blocking by default; drops messages if the queue is full.
        - Set block=True if you want back-pressure (non real-time contexts).

        This mirrors your ROS2 style: same data arguments, extra async controls.
        """
        self._bus.submit(self._publish_now, t, q, dq, ddq, block=block, timeout=timeout)


class DoubleArrayCsvPublisher:
    """
    CSV-based analogue of your ROS2 DoubleArrayPublisher.

    Instead of publishing a Float64MultiArray, it writes one row:

        [array...]

    You define the structure via `column_names`, e.g.:

        ["t", "x", "y", "z"]
    """

    def __init__(
        self,
        csv_path: str,
        column_names: str,
    ):
        """
        Parameters
        ----------
        csv_path : str
            Path to the CSV file to write to.
        column_names : Sequence[str]
            List of column names for the CSV.
            Must match the length of the array passed to publish_once.
        """
        self._writer = _CsvWriter(csv_path, column_names)
        self._bus = _get_global_bus()

    def _publish_now(self, t, array):
        """
        Synchronous implementation: immediately appends a row to the CSV.
        Runs in the background worker thread via the global bus.
        """
        if isinstance(array, np.ndarray):
            values = ndarray2list(array)
        else:
            values = _to_list(array)

        self._writer.append_row(t, values)

    def publish_once(
        self,
        t,
        array,
        *,
        block: bool = False,
        timeout: Optional[float] = None,
    ):
        """
        Enqueue a row [array...] to be written asynchronously.
        """
        self._bus.submit(self._publish_now, t,array, block=block, timeout=timeout)


class TestStartCsvPublisher:
    """
    CSV-based analogue of your ROS2 TestStartPublisher.

    Instead of publishing a std_msgs/Bool, this writes a row like:

        [bool_value]

    You decide the header, e.g. ["test_started"].
    """

    def __init__(
        self,
        csv_path: str,
        column_names: str,
    ):
        """
        Parameters
        ----------
        csv_path : str
            Path to the CSV file to write to.
        column_names : Sequence[str]
            Expected column names, e.g. ['test_started'] or
            ['timestamp', 'test_started'] (you can extend _publish_now if needed).
        """
        self._writer = _CsvWriter(csv_path, column_names)
        self._bus = _get_global_bus()

    def _publish_now(self, bool_value: bool):
        """
        Synchronous implementation: append a single bool row to the CSV.
        Runs on the background worker.
        """
        row = [bool(bool_value)]
        t = time()
        self._writer.append_row(t, row)

    def publish_once(
        self,
        bool_value: bool,
        *,
        block: bool = False,
        timeout: Optional[float] = None,
    ):
        """
        Enqueue a single Bool row to be written asynchronously.
        """
        self._bus.submit(self._publish_now, bool_value, block=block, timeout=timeout)


# -------------------------------------------------------------------
# Optional helper for one-shot write (convenience)
# -------------------------------------------------------------------

def append_test_start_once(
    csv_path: str,
    value: bool,
    column_names: Sequence[str] = ("test_started",),
):
    """
    Convenience function like your ROS2 `publish_test_start_once`, but for CSV.

    It just:
      - Ensures the CSV exists with the given header.
      - Appends a single row [value] synchronously (no async bus).
    """
    writer = _CsvWriter(csv_path, column_names)
    writer.append_row([bool(value)])



def swap_csv(path_in, path_out, index_0, index_1):
    df = pd.read_csv(path_in, header=0, index_col=False)
    suff_to_swap = ["x", "y", "z", "x_vel", "y_vel", "z_vel", "x_acc", "y_acc", "z_acc"]
    for suff in suff_to_swap:
        col_name0 = f"keypoint{index_0}_{suff}"
        col_name1 = f"keypoint{index_1}_{suff}"
        df[col_name0], df[col_name1] = df[col_name1].copy(), df[col_name0].copy()
    df.to_csv(path_out, header=True, index=False)


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------

# if __name__ == "__main__":
#     import time
#
#     # Example for JointTargetCsvPublisher
#     joint_logger = JointTargetCsvPublisher(
#         csv_path="joint_target.csv",
#         column_names=[
#             "q1", "q2", "q3",
#             "dq1", "dq2", "dq3",
#             "ddq1", "ddq2", "ddq3",
#         ],
#         joint_names=["joint1", "joint2", "joint3"],
#     )
#
#     # Example for DoubleArrayCsvPublisher
#     human_logger = DoubleArrayCsvPublisher(
#         csv_path="human_state.csv",
#         column_names=["t", "x", "y", "z"],
#     )
#
#     # Example for TestStartCsvPublisher
#     test_logger = TestStartCsvPublisher(
#         csv_path="test_start.csv",
#         column_names=["test_started"],
#     )
#
#     # Simulate some loop where you log stuff asynchronously
#     for i in range(10):
#         q = [0.0 + i * 0.01, 0.5, -0.3]
#         dq = [0.0, 0.0, 0.0]
#         ddq = [0.0, -0.1, 0.1]
#         joint_logger.publish_once(q, dq, ddq)
#
#         t_now = i * 0.1
#         human_state = [t_now, 1.0 + i, 2.0, 3.0]
#         human_logger.publish_once(human_state)
#
#         if i == 0:
#             test_logger.publish_once(True)
#
#         # Your main loop can keep running without being blocked by disk I/O
#         time.sleep(0.01)
#
#     # Optional: clean shutdown of the global bus
#     bus = _get_global_bus()
#     bus.shutdown(wait=True)
