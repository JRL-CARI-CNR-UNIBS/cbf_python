#!/usr/bin/env python3
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

from controller_manager_msgs.srv import SwitchController
from builtin_interfaces.msg import Duration as MsgDuration

from zed_skeleton_kinematics_msgs.msg import ObjectsKinematicsStamped

# NEW: import the abstract, ROS-agnostic base
from Command_bridge.base_command_bridge_abc import BaseCommandBridgeABC


class JointStateCommandBridge(Node, BaseCommandBridgeABC):
    """Bridge node: ROS2 JointState subscriber → command publisher + obstacles/kinematics."""

    def __init__(
        self,
        ordered_joint_names: Iterable[str],
        *,
        threshold: float = 0.05,
        node_name: str = "joint_state_command_bridge",
        joint_states_topic: str = "/joint_states",
        command_topic: str = "/forward_position_controller/commands",
        obstacles_topics: Iterable[str] = ("/rs1/poses", "/rs2/poses"),
        kinematics_topics: Iterable[str] = ("/zed/zed_node/body_trk/skeletons_kinematics",),
        start_executor: bool = True,
    ) -> None:
        # Ensure rclpy is initialized
        try:
            rclpy.get_default_context()
            if not rclpy.ok():
                rclpy.init()
        except Exception:
            try:
                rclpy.init()
            except Exception:
                pass

        Node.__init__(self, node_name)
        BaseCommandBridgeABC.__init__(self, ordered_joint_names, threshold=threshold)

        # ROS publisher/subscriber
        self._pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self._sub = self.create_subscription(
            JointState, joint_states_topic, self._on_joint_state_ros, qos_profile_sensor_data
        )

        # TF + caches
        self._tf_buffer: Buffer = Buffer()
        self._tf_listener: TransformListener = TransformListener(self._tf_buffer, self, spin_thread=False)
        self._frame_to_world_cache: Dict[str, np.ndarray] = {}
        self._last_tf_warn_time: Dict[str, float] = {}

        # Legacy PoseArray store (kept for compatibility)
        self.obstacles_: Dict[str, List[np.ndarray]] = {}
        self._obstacles_last_recv_: Dict[str, rclpy.time.Time] = {}
        self._poses_lock = threading.Lock()

        # Kinematics store: topic -> (pos[N,3], vel[N,3], acc[N,3])
        self.kinematics_: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._kin_last_recv_: Dict[str, rclpy.time.Time] = {}

        # Subscriptions for PoseArray
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

        # Subscriptions for kinematics
        self._kin_subs = []
        for topic in kinematics_topics:
            topic = str(topic)
            cb = partial(self._on_objects_kinematics, topic_name=topic)
            sub = self.create_subscription(ObjectsKinematicsStamped, topic, cb, qos_profile_sensor_data)
            self._kin_subs.append(sub)
            self.kinematics_[topic] = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))
            try:
                self._kin_last_recv_[topic] = rclpy.time.Time(seconds=0, nanoseconds=0)
            except Exception:
                self._kin_last_recv_[topic] = rclpy.time.Time()

        # Preallocated kinematics buffers per topic TO BE TESTED
        self._kin_buffers: Dict[str, Dict[str, np.ndarray]] = {}

        for topic in kinematics_topics:
            topic = str(topic)
            self._kin_buffers[topic] = {
                "pos": np.empty((0, 3), dtype=float),
                "vel": np.empty((0, 3), dtype=float),
                "acc": np.empty((0, 3), dtype=float),
            }

        # Optional executor
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

    # ---------------------------- ROS callbacks ----------------------------
    def _on_joint_state_ros(self, msg: JointState) -> None:
        # Delegate to transport-agnostic mapper
        self.map_joint_state(msg.name, msg.position, msg.velocity, msg.effort)

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

    # def _on_objects_kinematics(self, msg: ObjectsKinematicsStamped, *, topic_name: str) -> None:
    #     """Build (pos, vel, acc) from ObjectsKinematics and transform to 'world':contentReference[oaicite:1]{index=1}."""
    #     frame_id = (msg.header.frame_id or "").strip() or "world"
    #
    #     pos_list, vel_list, acc_list = [], [], [] TOBETESTED
    #
    #     for objkin in msg.objects:
    #         obj = getattr(objkin, "object", None)
    #         sk = getattr(obj, "skeleton_3d", None) if obj is not None else None
    #         if sk is None or not hasattr(sk, "keypoints"):
    #             continue
    #
    #         kps = sk.keypoints
    #         vels = getattr(objkin, "keypoint_velocities", []) or []
    #         accs = getattr(objkin, "keypoint_accelerations", []) or []
    #         n = min(len(kps), len(vels), len(accs))
    #         if n == 0:
    #             continue
    #
    #         for i in range(n):
    #             try:
    #                 px, py, pz = float(kps[i].kp[0]), float(kps[i].kp[1]), float(kps[i].kp[2])
    #             except Exception:
    #                 px = float(getattr(kps[i], "x", 0.0))
    #                 py = float(getattr(kps[i], "y", 0.0))
    #                 pz = float(getattr(kps[i], "z", 0.0))
    #             if abs(px) < 1e-12 and abs(py) < 1e-12 and abs(pz) < 1e-12:
    #                 continue
    #
    #             try:
    #                 vx, vy, vz = float(vels[i].kp[0]), float(vels[i].kp[1]), float(vels[i].kp[2])
    #             except Exception:
    #                 vx = float(getattr(vels[i], "x", 0.0))
    #                 vy = float(getattr(vels[i], "y", 0.0))
    #                 vz = float(getattr(vels[i], "z", 0.0))
    #
    #             try:
    #                 ax, ay, az = float(accs[i].kp[0]), float(accs[i].kp[1]), float(accs[i].kp[2])
    #             except Exception:
    #                 ax = float(getattr(accs[i], "x", 0.0))
    #                 ay = float(getattr(accs[i], "y", 0.0))
    #                 az = float(getattr(accs[i], "z", 0.0))
    #
    #             pos_list.append([px, py, pz])
    #             vel_list.append([vx, vy, vz])
    #             acc_list.append([ax, ay, az])
    #
    #     recv_time = self.get_clock().now()
    #     if not pos_list:
    #         with self._poses_lock:
    #             self._kin_last_recv_[topic_name] = recv_time
    #         return
    #
    #     pos_arr = np.asarray(pos_list, dtype=float)
    #     vel_arr = np.asarray(vel_list, dtype=float)
    #     acc_arr = np.asarray(acc_list, dtype=float)
    #
    #     if frame_id != "world":
    #         T = self._get_transform_matrix_to_world(frame_id, msg.header.stamp)
    #         if T is None:
    #             return
    #         R = T[:3, :3]
    #         t = T[:3, 3]
    #         pos_arr = (R @ pos_arr.T).T + t.reshape(1, 3)
    #         vel_arr = (R @ vel_arr.T).T
    #         acc_arr = (R @ acc_arr.T).T
    #
    #     with self._poses_lock:
    #         self.kinematics_[topic_name] = (pos_arr, vel_arr, acc_arr)
    #         self._kin_last_recv_[topic_name] = recv_time

    def _on_objects_kinematics(
            self,
            msg: ObjectsKinematicsStamped,
            *,
            topic_name: str
    ) -> None:
        """Efficient kinematics callback: vectorized, allocation-minimal."""

        frame_id = (msg.header.frame_id or "").strip() or "world"
        recv_time = self.get_clock().now()

        # -------- collect raw arrays (vectorized) -----------------------------
        pos_chunks = []
        vel_chunks = []
        acc_chunks = []

        for objkin in msg.objects:
            obj = getattr(objkin, "object", None)
            sk = getattr(obj, "skeleton_3d", None) if obj is not None else None
            if sk is None or not hasattr(sk, "keypoints"):
                continue

            kps = sk.keypoints
            vels = getattr(objkin, "keypoint_velocities", None)
            accs = getattr(objkin, "keypoint_accelerations", None)
            if not kps or not vels or not accs:
                continue

            n = min(len(kps), len(vels), len(accs))
            if n == 0:
                continue

            try:
                pos = np.asarray([kp.kp for kp in kps[:n]], dtype=float)
                vel = np.asarray([v.kp for v in vels[:n]], dtype=float)
                acc = np.asarray([a.kp for a in accs[:n]], dtype=float)
            except Exception:
                # Fallback for alternative field layouts
                pos = np.asarray([[kp.x, kp.y, kp.z] for kp in kps[:n]], dtype=float)
                vel = np.asarray([[v.x, v.y, v.z] for v in vels[:n]], dtype=float)
                acc = np.asarray([[a.x, a.y, a.z] for a in accs[:n]], dtype=float)

            # Filter invalid (zero) keypoints in one shot
            mask = np.linalg.norm(pos, axis=1) > 1e-12
            if not np.any(mask):
                continue

            pos_chunks.append(pos[mask])
            vel_chunks.append(vel[mask])
            acc_chunks.append(acc[mask])

        # -------- no data case -------------------------------------------------
        if not pos_chunks:
            with self._poses_lock:
                self._kin_last_recv_[topic_name] = recv_time
            return

        # -------- concatenate once --------------------------------------------
        pos_all = np.concatenate(pos_chunks, axis=0)
        vel_all = np.concatenate(vel_chunks, axis=0)
        acc_all = np.concatenate(acc_chunks, axis=0)

        # -------- transform to world (vectorized) ------------------------------
        if frame_id != "world":
            T = self._get_transform_matrix_to_world(frame_id, msg.header.stamp)
            if T is None:
                return

            R = T[:3, :3]
            t = T[:3, 3]

            pos_all = pos_all @ R.T + t
            vel_all = vel_all @ R.T
            acc_all = acc_all @ R.T

        # -------- reuse preallocated buffers ----------------------------------
        buf = self._kin_buffers[topic_name]
        n = pos_all.shape[0]

        if buf["pos"].shape[0] < n:
            buf["pos"] = np.empty((n, 3), dtype=float)
            buf["vel"] = np.empty((n, 3), dtype=float)
            buf["acc"] = np.empty((n, 3), dtype=float)

        buf["pos"][:n] = pos_all
        buf["vel"][:n] = vel_all
        buf["acc"][:n] = acc_all

        # -------- publish atomically ------------------------------------------
        with self._poses_lock:
            self.kinematics_[topic_name] = (
                buf["pos"][:n],
                buf["vel"][:n],
                buf["acc"][:n],
            )
            self._kin_last_recv_[topic_name] = recv_time

    # ---------------------------- ABC overrides ----------------------------
    def _do_publish(self, q: np.ndarray) -> None:
        msg = Float64MultiArray()
        msg.data = q.tolist()
        self._pub.publish(msg)

    def getObstacles(self,elapsed  = 0.0,  max_age_sec: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate recent kinematics into (pos, vel, acc) as in the original node:contentReference[oaicite:2]{index=2}."""
        now = self.get_clock().now()
        pos_all, vel_all, acc_all = [], [], []

        with self._poses_lock:
            for topic, triple in self.kinematics_.items():
                last = self._kin_last_recv_.get(topic)
                if last is None:
                    continue
                try:
                    age_sec = float((now - last).nanoseconds) * 1e-9
                except Exception:
                    age_sec = float("inf")
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

    # ---------------------------- TF + utilities (ROS-specific) ----------------------------
    def _get_transform_matrix_to_world(self, frame_id: str, stamp) -> Optional[np.ndarray]:
        if frame_id in self._frame_to_world_cache:
            return self._frame_to_world_cache[frame_id]
        try:
            time_obj = rclpy.time.Time(
                seconds=getattr(stamp, "sec", 0), nanoseconds=getattr(stamp, "nanosec", 0)
            )
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
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=float,
        )

    def shutdown(self) -> None:
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

    def switch_to_forward_position_controller_service(self, timeout_sec: float = 10.0) -> None:
        """Stop trajectory controller and start forward position controller, as before:contentReference[oaicite:3]{index=3}."""
        client = self.create_client(SwitchController, "/controller_manager/switch_controller")
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("/controller_manager/switch_controller service not available")

        req = SwitchController.Request()
        req.activate_controllers = []
        req.deactivate_controllers = []
        req.start_controllers = ["forward_position_controller"]
        req.stop_controllers = ["scaled_joint_trajectory_controller"]

        if hasattr(req, "strictness"):
            req.strictness = 0
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
