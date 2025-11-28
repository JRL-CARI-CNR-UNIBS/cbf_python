from __future__ import annotations

import threading
from typing import Iterable, List, Optional, Dict

import numpy as np
import time
from functools import partial

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray
from tf2_ros import Buffer, TransformListener

import math
from controller_manager_msgs.srv import SwitchController
from builtin_interfaces.msg import Duration as MsgDuration

# ZED skeletons
from zed_msgs.msg import ObjectsStamped


class JointStateCommandBridge(Node):
    """Bridge node: JointState subscriber → Float64MultiArray publisher + obstacles."""

    def __init__(
        self,
        ordered_joint_names: Iterable[str],
        *,
        threshold: float = 0.05,
        node_name: str = "joint_state_command_bridge",
        joint_states_topic: str = "/joint_states",
        command_topic: str = "/forward_position_controller/commands",
        obstacles_topics: Iterable[str] = ("/rs1/poses", "/rs2/poses"),
        skeleton_topics: Iterable[str] = ("/zed/zed_node/body_trk/skeletons",),
        start_executor: bool = True,
        publish_period_sec: float = 0.002,     # NEW: Tc (publisher thread period)
    ) -> None:
        # Initialize rclpy context if not already done
        try:
            rclpy.get_default_context()
            if not rclpy.ok():
                rclpy.init()
        except Exception:
            try:
                rclpy.init()
            except Exception:
                pass

        super().__init__(node_name)

        self.ordered_joint_names_: List[str] = list(ordered_joint_names)
        self.threshold: float = float(threshold)

        n = len(self.ordered_joint_names_)
        self.actual_joint_positions_ = np.full(n, np.nan, dtype=float)
        self.actual_joint_velocities_ = np.full(n, np.nan, dtype=float)
        self.actual_joint_efforts_ = np.full(n, np.nan, dtype=float)
        self._state_lock = threading.Lock()

        # Publisher (queue depth 10 is fine for commands)
        self._pub = self.create_publisher(Float64MultiArray, command_topic, 10)

        # Subscriber: use sensor-data QoS for low-latency best-effort
        self._sub = self.create_subscription(
            JointState, joint_states_topic, self._on_joint_state, qos_profile_sensor_data
        )

        # --- TF and obstacles subscribers ---
        self._tf_buffer: Buffer = Buffer()
        self._tf_listener: TransformListener = TransformListener(self._tf_buffer, self, spin_thread=False)
        self._frame_to_world_cache: Dict[str, np.ndarray] = {}
        self._last_tf_warn_time: Dict[str, float] = {}

        # Per-topic obstacles and last receive times
        self.obstacles_: Dict[str, List[np.ndarray]] = {}
        self._obstacles_last_recv_: Dict[str, rclpy.time.Time] = {}
        self._poses_lock = threading.Lock()

        # Create one subscription per PoseArray topic
        self._poses_subs = []
        for topic in obstacles_topics:
            topic = str(topic)
            cb = partial(self._on_pose_array, topic_name=topic)
            sub = self.create_subscription(PoseArray, topic, cb, qos_profile_sensor_data)
            self._poses_subs.append(sub)
            self.obstacles_[topic] = []
            try:
                self._obstacles_last_recv_[topic] = rclpy.time.Time(seconds=0, nanoseconds=0)
            except Exception:
                self._obstacles_last_recv_[topic] = rclpy.time.Time()

        # Create one subscription per ZED ObjectsStamped topic (skeletons)
        self._skeleton_subs = []
        for topic in skeleton_topics:
            topic = str(topic)
            cb = partial(self._on_objects_stamped, topic_name=topic)
            sub = self.create_subscription(ObjectsStamped, topic, cb, qos_profile_sensor_data)
            self._skeleton_subs.append(sub)
            if topic not in self.obstacles_:
                self.obstacles_[topic] = []
            if topic not in self._obstacles_last_recv_:
                try:
                    self._obstacles_last_recv_[topic] = rclpy.time.Time(seconds=0, nanoseconds=0)
                except Exception:
                    self._obstacles_last_recv_[topic] = rclpy.time.Time()

        # Optional internal executor thread
        self._executor: Optional[SingleThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None
        if start_executor:
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self)
            self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._spin_thread.start()

        # --- NEW: command storage + publishing thread state ---
        self._cmd_lock = threading.Lock()
        self._cmd_q = np.zeros(n, dtype=float)
        self._cmd_qp = np.zeros(n, dtype=float)
        self._cmd_qpp = np.zeros(n, dtype=float)
        self._new_command = False
        self._Tc = float(publish_period_sec)
        self._pub_stop = threading.Event()
        self._publishing_thread = threading.Thread(target=self._publishing_loop, name="publishing_thread", daemon=True)
        self._publishing_thread.start()

        self.get_logger().info(
            f"Initialized with joints: {self.ordered_joint_names_}; "
            f"threshold={self.threshold}; "
            f"obstacles_topics={list(obstacles_topics)}; "
            f"skeleton_topics={list(skeleton_topics)}; "
            f"Tc={self._Tc}s"
        )

    # ---------------------------- Callbacks ----------------------------
    def _on_joint_state(self, msg: JointState) -> None:
        name_to_idx = {name: i for i, name in enumerate(msg.name)}

        pos = np.full_like(self.actual_joint_positions_, np.nan)
        vel = np.full_like(self.actual_joint_velocities_, np.nan)
        eff = np.full_like(self.actual_joint_efforts_, np.nan)

        for j, joint in enumerate(self.ordered_joint_names_):
            idx = name_to_idx.get(joint)
            if idx is None:
                continue
            if idx < len(msg.position):
                pos[j] = float(msg.position[idx])
            if idx < len(msg.velocity):
                vel[j] = float(msg.velocity[idx])
            if idx < len(msg.effort):
                eff[j] = float(msg.effort[idx])

        with self._state_lock:
            self.actual_joint_positions_[:] = pos
            self.actual_joint_velocities_[:] = vel
            self.actual_joint_efforts_[:] = eff

    def _on_pose_array(self, msg: PoseArray, *, topic_name: str) -> None:
        frame_id = (msg.header.frame_id or "").strip() or "world"
        pts = [
            np.array([float(p.position.x), float(p.position.y), float(p.position.z)], dtype=float)
            for p in msg.poses
        ]
        if frame_id != "world":
            T = self._get_transform_matrix_to_world(frame_id, msg.header.stamp)
            if T is None:
                return
            pts_world = [(T @ np.array([v[0], v[1], v[2], 1.0], dtype=float))[:3] for v in pts]
        else:
            pts_world = pts

        recv_time = self.get_clock().now()
        with self._poses_lock:
            self.obstacles_[topic_name] = [np.asarray(v, dtype=float) for v in pts_world]
            self._obstacles_last_recv_[topic_name] = recv_time

    def _on_objects_stamped(self, msg: ObjectsStamped, *, topic_name: str) -> None:
        frame_id = (msg.header.frame_id or "").strip() or "world"
        raw_pts: List[np.ndarray] = []
        try:
            objects = msg.objects
        except Exception:
            objects = []

        for obj in objects:
            sk = getattr(obj, "skeleton_3d", None)
            if sk is None:
                continue
            kps = getattr(sk, "keypoints", None)
            if not kps:
                continue
            for kp in kps:
                xyz = getattr(kp, "kp", kp)
                if xyz is None or len(xyz) < 3:
                    continue
                x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
                if (x == 0.0) and (y == 0.0) and (z == 0.0):
                    continue
                raw_pts.append(np.array([x, y, z], dtype=float))

        if not raw_pts:
            recv_time = self.get_clock().now()
            with self._poses_lock:
                self._obstacles_last_recv_[topic_name] = recv_time
            return

        if frame_id != "world":
            T = self._get_transform_matrix_to_world(frame_id, msg.header.stamp)
            if T is None:
                return
            pts_world = [(T @ np.array([p[0], p[1], p[2], 1.0], dtype=float))[:3] for p in raw_pts]
        else:
            pts_world = raw_pts

        recv_time = self.get_clock().now()
        with self._poses_lock:
            self.obstacles_[topic_name] = [np.asarray(v, dtype=float) for v in pts_world]
            self._obstacles_last_recv_[topic_name] = recv_time

    # ---------------------------- Commands ----------------------------
    def sendCommand(self, q: np.ndarray, qp: np.ndarray, qpp: np.ndarray) -> None:
        """
        Store the desired (q, qp, qpp) for publishing thread.
        Raises ValueError if max(abs(q - current_positions)) > threshold (ignoring NaNs),
        or if vector sizes mismatch.
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        qp = np.asarray(qp, dtype=float).reshape(-1)
        qpp = np.asarray(qpp, dtype=float).reshape(-1)

        n = len(self.ordered_joint_names_)
        if q.size != n or qp.size != n or qpp.size != n:
            raise ValueError(f"Expected vectors of length {n}; got q={q.size}, qp={qp.size}, qpp={qpp.size}")

        # threshold check versus latest actual positions (ignore NaNs)
        with self._state_lock:
            curr = self.actual_joint_positions_.copy()
        mask = ~np.isnan(curr)
        max_diff = float(np.max(np.abs(q[mask] - curr[mask]))) if np.any(mask) else 0.0
        if max_diff > self.threshold:
            raise ValueError(
                f"Command difference {max_diff:.3f} exceeds threshold {self.threshold:.3f}"
            )

        # store and mark new command
        with self._cmd_lock:
            self._cmd_q[:] = q
            self._cmd_qp[:] = qp
            self._cmd_qpp[:] = qpp
            self._new_command = True  # external thread notified

    # ---------------------------- Publishing loop (NEW) ----------------------------
    def _publishing_loop(self) -> None:
        Tc = self._Tc
        next_t = time.monotonic()
        while not self._pub_stop.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.001, next_t - now))
                continue
            next_t += Tc

            # -------- first short lock: snapshot + clear-once if new ----------
            with self._cmd_lock:
                q_snapshot = self._cmd_q.copy()
                qp_snapshot = self._cmd_qp.copy()
                qpp_snapshot = self._cmd_qpp.copy()
                saw_new = self._new_command
                if saw_new:
                    # consume the new-command edge; publish snapshot as-is
                    self._new_command = False

            if saw_new:
                # publish outside the lock
                msg = Float64MultiArray()
                msg.data = q_snapshot.tolist()
                self._pub.publish(msg)
                continue

            # -------- no new command → integrate outside the lock -------------
            q_int = q_snapshot + qp_snapshot * Tc + 0.5 * qpp_snapshot * (Tc ** 2)
            qp_int = qp_snapshot + qpp_snapshot * Tc

            # -------- second short lock: check for races + write-back ---------
            publish_data = None
            with self._cmd_lock:
                if self._new_command:
                    # A new command arrived during integration → publish the freshest q
                    publish_data = self._cmd_q.copy()
                    self._new_command = False
                else:
                    # Commit integrated state and publish it
                    self._cmd_q[:] = q_int
                    self._cmd_qp[:] = qp_int
                    publish_data = q_int

            # publish outside the lock
            msg = Float64MultiArray()
            msg.data = publish_data.tolist()
            self._pub.publish(msg)

    # ---------------------------- TF helpers ----------------------------
    def _get_transform_matrix_to_world(self, frame_id: str, stamp) -> Optional[np.ndarray]:
        if frame_id in self._frame_to_world_cache:
            return self._frame_to_world_cache[frame_id]
        try:
            time_obj = rclpy.time.Time(seconds=getattr(stamp, 'sec', 0), nanoseconds=getattr(stamp, 'nanosec', 0))
        except Exception:
            time_obj = rclpy.time.Time()
        try:
            ts = self._tf_buffer.lookup_transform("world", frame_id, time_obj)
        except Exception as e:
            last = self._last_tf_warn_time.get(frame_id, 0.0)
            now = time.monotonic()
            if now - last > 2.0:
                self.get_logger().warn(f"TF to world unavailable for '{frame_id}': {e}")
                self._last_tf_warn_time[frame_id] = now
            return None

        t = ts.transform.translation
        q = ts.transform.rotation
        T = np.eye(4, dtype=float)
        R = self._quat_to_rot(q.x, q.y, q.z, q.w)
        T[:3, :3] = R
        T[:3, 3] = np.array([t.x, t.y, t.z], dtype=float)
        self._frame_to_world_cache[frame_id] = T
        return T

    @staticmethod
    def _quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
        q = np.array([x, y, z, w], dtype=float)
        n = float(np.linalg.norm(q))
        if n == 0.0:
            return np.eye(3, dtype=float)
        x, y, z, w = q / n
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        return np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
            [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
            [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)],
        ], dtype=float)

    # ---------------------------- Getters ----------------------------
    def getPositions(self) -> np.ndarray:
        with self._state_lock:
            return self.actual_joint_positions_.copy()

    def getVelocities(self) -> np.ndarray:
        with self._state_lock:
            return self.actual_joint_velocities_.copy()

    def getEfforts(self) -> np.ndarray:
        with self._state_lock:
            return self.actual_joint_efforts_.copy()

    def _index_of(self, name: str) -> int:
        try:
            return self.ordered_joint_names_.index(name)
        except ValueError as e:
            raise KeyError(f"Unknown joint name: {name}") from e

    def getJointPosition(self, name: str) -> float:
        idx = self._index_of(name)
        with self._state_lock:
            return float(self.actual_joint_positions_[idx])

    def getJointVelocity(self, name: str) -> float:
        idx = self._index_of(name)
        with self._state_lock:
            return float(self.actual_joint_velocities_[idx])

    def getJointEffort(self, name: str) -> float:
        idx = self._index_of(name)
        with self._state_lock:
            return float(self.actual_joint_efforts_[idx])

    def getObstaclesPoses(self, max_age_sec: float = 0.5) -> List[np.ndarray]:
        now = self.get_clock().now()
        combined: List[np.ndarray] = []
        with self._poses_lock:
            for topic, poses in self.obstacles_.items():
                last = self._obstacles_last_recv_.get(topic)
                if last is None:
                    continue
                age_sec = float((now - last).nanoseconds) * 1e-9
                if age_sec <= float(max_age_sec):
                    combined.extend([v.copy() for v in poses])
        return combined

    # ---------------------------- Shutdown ----------------------------
    def shutdown(self) -> None:
        """Stop threads/executor and destroy the node."""
        # stop publishing thread
        self._pub_stop.set()
        try:
            if self._publishing_thread is not None:
                self._publishing_thread.join(timeout=1.0)
        except Exception:
            pass

        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:
                pass
            try:
                self._executor.remove_node(self)
            except Exception:
                pass
            self._executor = None

        self.destroy_node()
        # Leave rclpy.shutdown() to the app/main.
