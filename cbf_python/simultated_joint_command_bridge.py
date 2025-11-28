#!/usr/bin/env python3
"""
JointStateCommandBridge (ROS 2 Humble, Python)

Versione modificata per leggere i **zed_skeleton_kinematics_msgs/ObjectsKinematicsStamped**
e fornire il metodo:
    getObstacles(max_age_sec=0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
che restituisce (pos, vel, acc) come array NumPy [N,3] ciascuno, aggregando i
keypoint validi entro una certa età massima.

Note:
- I keypoint con posizione [0,0,0] vengono scartati.
- Le posizioni sono trasformate nel frame 'world' usando TF. Le velocità/accelerazioni
  sono trasformate solo con la rotazione (niente traslazione), perché sono vettori.
- Mantiene la logica di base di sottoscrizione a /joint_states e invio comandi come nell'originale【27†source】.
"""

from __future__ import annotations

import threading
from typing import Iterable, List, Optional, Dict, Tuple

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

# Messaggio dei kinematics (nuovo)
from zed_skeleton_kinematics_msgs.msg import ObjectsKinematicsStamped


class JointStateCommandBridge(Node):
    """Bridge node: JointState subscriber → Float64MultiArray publisher + obstacles/kinematics."""

    def __init__(
        self,
        ordered_joint_names: Iterable[str],
        *,
        threshold: float = 0.05,
        node_name: str = "joint_state_command_bridge",
        joint_states_topic: str = "/joint_states",
        command_topic: str = "/forward_position_controller/commands",
        obstacles_topics: Iterable[str] = ("/rs1/poses", "/rs2/poses"),
        # Nuovo: topic con ObjectsKinematicsStamped pubblicati dal nodo kinematics
        kinematics_topics: Iterable[str] = ("/zed/zed_node/body_trk/skeletons_kinematics",),
        start_executor: bool = True,
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

        # --- TF and subscribers ---
        self._tf_buffer: Buffer = Buffer()
        self._tf_listener: TransformListener = TransformListener(self._tf_buffer, self, spin_thread=False)
        self._frame_to_world_cache: Dict[str, np.ndarray] = {}
        self._last_tf_warn_time: Dict[str, float] = {}

        # Storage per PoseArray (posizioni) - rimane per compatibilità
        self.obstacles_: Dict[str, List[np.ndarray]] = {}
        self._obstacles_last_recv_: Dict[str, rclpy.time.Time] = {}
        self._poses_lock = threading.Lock()

        # Storage per Kinematics: topic -> (pos[N,3], vel[N,3], acc[N,3])
        self.kinematics_: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._kin_last_recv_: Dict[str, rclpy.time.Time] = {}

        # Create subscriptions per PoseArray topic
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

        # Create subscriptions per kinematics topic (ObjectsKinematicsStamped)
        self._kin_subs = []
        for topic in kinematics_topics:
            topic = str(topic)
            cb = partial(self._on_objects_kinematics, topic_name=topic)
            sub = self.create_subscription(ObjectsKinematicsStamped, topic, cb, qos_profile_sensor_data)
            self._kin_subs.append(sub)
            self.kinematics_[topic] = (np.zeros((0,3)), np.zeros((0,3)), np.zeros((0,3)))
            try:
                self._kin_last_recv_[topic] = rclpy.time.Time(seconds=0, nanoseconds=0)
            except Exception:
                self._kin_last_recv_[topic] = rclpy.time.Time()

        # Optional internal executor thread
        self._executor: Optional[SingleThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None
        if start_executor:
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self)
            self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._spin_thread.start()

        self.get_logger().info(
            f"Initialized with joints: {self.ordered_joint_names_}; "
            f"threshold={self.threshold}; "
            f"obstacles_topics={list(obstacles_topics)}; "
            f"kinematics_topics={list(kinematics_topics)}"
        )

    # ---------------------------- Callbacks ----------------------------
    def _on_joint_state(self, msg: JointState) -> None:
        """Subscription callback: map incoming state to our ordered arrays."""
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
        """Callback per PoseArray: salva posizioni in world frame e timestamp (compatibilità)."""
        frame_id = (msg.header.frame_id or "").strip() or "world"

        pts = [
            np.array([float(p.position.x), float(p.position.y), float(p.position.z)], dtype=float)
            for p in msg.poses
        ]

        # Transform to world
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

    def _on_objects_kinematics(self, msg: ObjectsKinematicsStamped, *, topic_name: str) -> None:
        """Callback per ObjectsKinematicsStamped: costruisce (pos, vel, acc) e li porta in 'world'."""
        frame_id = (msg.header.frame_id or "").strip() or "world"

        pos_list, vel_list, acc_list = [], [], []

        for objkin in msg.objects:
            obj = getattr(objkin, "object", None)
            sk = getattr(obj, "skeleton_3d", None) if obj is not None else None
            if sk is None or not hasattr(sk, "keypoints"):
                continue

            kps = sk.keypoints
            vels = getattr(objkin, "keypoint_velocities", []) or []
            accs = getattr(objkin, "keypoint_accelerations", []) or []

            n = min(len(kps), len(vels), len(accs))
            if n == 0:
                continue

            for i in range(n):
                # posizione keypoint
                try:
                    px, py, pz = float(kps[i].kp[0]), float(kps[i].kp[1]), float(kps[i].kp[2])
                except Exception:
                    # fallback per eventuali strutture diverse
                    px = float(getattr(kps[i], "x", 0.0))
                    py = float(getattr(kps[i], "y", 0.0))
                    pz = float(getattr(kps[i], "z", 0.0))

                # Scarta keypoint nulli
                if abs(px) < 1e-12 and abs(py) < 1e-12 and abs(pz) < 1e-12:
                    continue

                # velocità
                try:
                    vx, vy, vz = float(vels[i].kp[0]), float(vels[i].kp[1]), float(vels[i].kp[2])
                except Exception:
                    vx = float(getattr(vels[i], "x", 0.0))
                    vy = float(getattr(vels[i], "y", 0.0))
                    vz = float(getattr(vels[i], "z", 0.0))

                # accelerazione
                try:
                    ax, ay, az = float(accs[i].kp[0]), float(accs[i].kp[1]), float(accs[i].kp[2])
                except Exception:
                    ax = float(getattr(accs[i], "x", 0.0))
                    ay = float(getattr(accs[i], "y", 0.0))
                    az = float(getattr(accs[i], "z", 0.0))

                pos_list.append([px, py, pz])
                vel_list.append([vx, vy, vz])
                acc_list.append([ax, ay, az])

        if not pos_list:
            # Nessun dato valido: non sovrascrivo l'ultimo stato
            recv_time = self.get_clock().now()
            with self._poses_lock:
                self._kin_last_recv_[topic_name] = recv_time
            return

        pos_arr = np.asarray(pos_list, dtype=float)
        vel_arr = np.asarray(vel_list, dtype=float)
        acc_arr = np.asarray(acc_list, dtype=float)

        # Trasformazioni al frame 'world':
        # - posizioni: T (R,t)
        # - velocità/accelerazioni: solo R
        if frame_id != "world":
            T = self._get_transform_matrix_to_world(frame_id, msg.header.stamp)
            if T is None:
                return
            R = T[:3, :3]
            t = T[:3, 3]
            # pos
            pos_arr = (R @ pos_arr.T).T + t.reshape(1, 3)
            # vel/acc
            vel_arr = (R @ vel_arr.T).T
            acc_arr = (R @ acc_arr.T).T

        recv_time = self.get_clock().now()
        with self._poses_lock:
            self.kinematics_[topic_name] = (pos_arr, vel_arr, acc_arr)
            self._kin_last_recv_[topic_name] = recv_time

    # ---------------------------- Commands ----------------------------
    def sendCommand(self, q: np.ndarray) -> None:
        """Publish a Float64MultiArray command if within the allowed difference.

        Raises a ValueError if `max(abs(q - current_positions)) > threshold`.
        """
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        if q_arr.size != len(self.ordered_joint_names_):
            raise ValueError(
                f"q has length {q_arr.size}, but expected {len(self.ordered_joint_names_)}"
            )

        with self._state_lock:
            curr = self.actual_joint_positions_.copy()

        mask = ~np.isnan(curr)
        max_diff = float(np.max(np.abs(q_arr[mask] - curr[mask]))) if np.any(mask) else 0.0

        if max_diff > self.threshold:
            raise ValueError(
                f"Command difference {max_diff:.3f} exceeds threshold {self.threshold:.3f}"
            )

        msg = Float64MultiArray()
        msg.data = q_arr.tolist()
        self._pub.publish(msg)

    # ---------------------------- TF helpers ----------------------------
    def _get_transform_matrix_to_world(self, frame_id: str, stamp) -> Optional[np.ndarray]:
        """Return a 4x4 transform matrix from `frame_id` to 'world'. Cache results.
        Returns None if no transform is available.
        """
        if frame_id in self._frame_to_world_cache:
            return self._frame_to_world_cache[frame_id]
        # Build a rclpy Time if stamp is present; else use latest
        try:
            time_obj = rclpy.time.Time(seconds=getattr(stamp, 'sec', 0), nanoseconds=getattr(stamp, 'nanosec', 0))
        except Exception:
            time_obj = rclpy.time.Time()
        try:
            ts = self._tf_buffer.lookup_transform("world", frame_id, time_obj)
        except Exception as e:
            # Throttle warnings per frame
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
        """Quaternion (x,y,z,w) → 3x3 rotation matrix."""
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

    # >>> Nuovo metodo richiesto
    def getObstacles(self, max_age_sec: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ritorna tuple (pos, vel, acc) come np.ndarray[N,3] ciascuno,
        aggregando tutti i topic kinematics con ultimo messaggio entro `max_age_sec`.
        """
        now = self.get_clock().now()
        pos_all, vel_all, acc_all = [], [], []

        with self._poses_lock:
            for topic, triple in self.kinematics_.items():
                last = self._kin_last_recv_.get(topic)
                if last is None:
                    continue
                # Calcola età
                try:
                    age_sec = float((now - last).nanoseconds) * 1e-9
                except Exception:
                    age_sec = float('inf')
                if age_sec <= float(max_age_sec):
                    p, v, a = triple
                    if p.size:
                        pos_all.append(p)
                        vel_all.append(v)
                        acc_all.append(a)

        if pos_all:
            return (np.vstack(pos_all), np.vstack(vel_all), np.vstack(acc_all))
        else:
            z = np.zeros((0, 3), dtype=float)
            return (z, z.copy(), z.copy())

    # ---------------------------- Shutdown ----------------------------
    def shutdown(self) -> None:
        """Stop the internal executor (if used) and destroy the node."""
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
        # Do not call rclpy.shutdown() here; leave it to the app/main.

    # ---------------------------- Utilities ----------------------------
    def wait_for_first_state(self, joint_name: str, timeout: float = 5.0) -> float:
        """Wait until we have a non-NaN position for `joint_name` or timeout.
        Returns the (possibly NaN) position after waiting.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            val = self.getJointPosition(joint_name)
            if not math.isnan(val):
                return val
            time.sleep(0.02)
        return self.getJointPosition(joint_name)

    def switch_to_forward_position_controller_service(self, timeout_sec: float = 10.0) -> None:
        """Call /controller_manager/switch_controller to stop/start controllers.

        Stops `scaled_joint_trajectory_controller` and starts
        `forward_position_controller` using the official service API.
        """
        client = self.create_client(SwitchController, "/controller_manager/switch_controller")
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("/controller_manager/switch_controller service not available")

        req = SwitchController.Request()
        req.activate_controllers = []
        req.deactivate_controllers = []
        req.start_controllers = ["forward_position_controller"]
        req.stop_controllers = ["scaled_joint_trajectory_controller"]

        if hasattr(req, "strictness"):
            req.strictness = 0  # BEST_EFFORT in many ros2_control versions
        if hasattr(req, "start_asap"):
            req.start_asap = False
        if hasattr(req, "activate_asap"):
            req.activate_asap = False
        if hasattr(req, "timeout"):
            req.timeout = MsgDuration(sec=0, nanosec=0)

        future = client.call_async(req)

        if self._executor is None:
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        else:
            deadline = time.monotonic() + timeout_sec if timeout_sec is not None else None
            while not future.done() and (deadline is None or time.monotonic() < deadline):
                time.sleep(0.01)

        if not future.done():
            raise TimeoutError("switch_controller call timed out")

        resp = future.result()
        ok = getattr(resp, "ok", True)
        if not ok:
            raise RuntimeError("switch_controller returned ok=False")

        self.get_logger().info("Controller switch request completed successfully.")


# Quick test harness (optional)
# if __name__ == "__main__":
#     rclpy.init()
#     bridge = JointStateCommandBridge(
#         ["joint1", "joint2"],
#         kinematics_topics=["/zed/zed_node/body_trk/skeletons_kinematics"],
#     )
#     try:
#         import time as _time
#         t0 = _time.time()
#         while _time.time() - t0 < 3.0:
#             _time.sleep(0.1)
#         pos, vel, acc = bridge.getObstacles(0.5)
#         print(pos.shape, vel.shape, acc.shape)
#     finally:
#         bridge.shutdown()
#         rclpy.shutdown()
