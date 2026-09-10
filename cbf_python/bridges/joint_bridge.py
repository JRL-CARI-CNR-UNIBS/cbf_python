"""
ROS 2 JointState and ForwardPosition command bridge with TF obstacle integration.
"""

from __future__ import annotations
import threading
import time
from typing import Iterable, List, Optional, Dict, Tuple
from functools import partial

import numpy as np
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

try:
    from zed_skeleton_kinematics_msgs.msg import ObjectsKinematicsStamped
    HAS_ZED_KINEMATICS = True
except ImportError:
    HAS_ZED_KINEMATICS = False

from cbf_python.bridges.base_bridge import BaseCommandBridgeABC


class JointStateCommandBridge(Node, BaseCommandBridgeABC):
    """Bridge node connecting ROS 2 joint states, controller commands, and TF obstacle tracking."""

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

        self._pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self._sub = self.create_subscription(
            JointState, joint_states_topic, self._on_joint_state_ros, qos_profile_sensor_data
        )

        self._tf_buffer: Buffer = Buffer()
        self._tf_listener: TransformListener = TransformListener(self._tf_buffer, self, spin_thread=False)
        self._frame_to_world_cache: Dict[str, np.ndarray] = {}
        self._last_tf_warn_time: Dict[str, float] = {}

        self.obstacles_: Dict[str, List[np.ndarray]] = {}
        self._obstacles_last_recv_: Dict[str, rclpy.time.Time] = {}
        self._poses_lock = threading.Lock()

        self.kinematics_: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._kin_last_recv_: Dict[str, rclpy.time.Time] = {}

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

        self._kin_subs = []
        if HAS_ZED_KINEMATICS:
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

        self._kin_buffers: Dict[str, Dict[str, np.ndarray]] = {}
        for topic in kinematics_topics:
            topic = str(topic)
            self._kin_buffers[topic] = {
                "pos": np.empty((0, 3), dtype=float),
                "vel": np.empty((0, 3), dtype=float),
                "acc": np.empty((0, 3), dtype=float),
            }

        self._executor: Optional[SingleThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None
        if start_executor:
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self)
            self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._spin_thread.start()

    def _do_publish(self, q: np.ndarray) -> None:
        msg = Float64MultiArray()
        msg.data = [float(x) for x in q]
        self._pub.publish(msg)

    def _on_joint_state_ros(self, msg: JointState) -> None:
        self.map_joint_state(
            msg.name,
            msg.position,
            msg.velocity if len(msg.velocity) else None,
            msg.effort if len(msg.effort) else None,
        )

    def _on_pose_array(self, msg: PoseArray, topic_name: str) -> None:
        frame_id = msg.header.frame_id or "world"
        stamp = msg.header.stamp
        T_fw = self._get_transform_matrix_to_world(frame_id, stamp)
        new_poses: List[np.ndarray] = []

        if T_fw is None:
            R = np.eye(3)
            p = np.zeros(3)
        else:
            R = T_fw[:3, :3]
            p = T_fw[:3, 3]

        for pose in msg.poses:
            p_local = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
            p_world = R @ p_local + p
            new_poses.append(p_world)

        with self._poses_lock:
            self.obstacles_[topic_name] = new_poses
            self._obstacles_last_recv_[topic_name] = self.get_clock().now()

    def _on_objects_kinematics(self, msg: Any, topic_name: str) -> None:
        frame_id = msg.header.frame_id or "world"
        stamp = msg.header.stamp
        T_fw = self._get_transform_matrix_to_world(frame_id, stamp)

        if T_fw is None:
            R = np.eye(3)
            p = np.zeros(3)
        else:
            R = T_fw[:3, :3]
            p = T_fw[:3, 3]

        total_kps = sum(len(getattr(obj, "keypoints_kinematics", [])) for obj in msg.objects)
        buf = self._kin_buffers.get(topic_name)
        if buf is None or buf["pos"].shape[0] < total_kps:
            buf = {
                "pos": np.empty((total_kps, 3), dtype=float),
                "vel": np.empty((total_kps, 3), dtype=float),
                "acc": np.empty((total_kps, 3), dtype=float),
            }
            self._kin_buffers[topic_name] = buf

        pos_buf = buf["pos"]
        vel_buf = buf["vel"]
        acc_buf = buf["acc"]
        k_valid = 0

        for obj in msg.objects:
            kps = getattr(obj, "keypoints_kinematics", None)
            if not kps:
                continue
            for kp in kps:
                pos_local = getattr(kp, "position", None)
                if pos_local is None:
                    continue
                px = float(getattr(pos_local, "x", 0.0))
                py = float(getattr(pos_local, "y", 0.0))
                pz = float(getattr(pos_local, "z", 0.0))
                if px == 0.0 and py == 0.0 and pz == 0.0:
                    continue

                vel_local = getattr(kp, "velocity", None)
                vx = float(getattr(vel_local, "x", 0.0)) if vel_local is not None else 0.0
                vy = float(getattr(vel_local, "y", 0.0)) if vel_local is not None else 0.0
                vz = float(getattr(vel_local, "z", 0.0)) if vel_local is not None else 0.0

                acc_local = getattr(kp, "acceleration", None)
                ax = float(getattr(acc_local, "x", 0.0)) if acc_local is not None else 0.0
                ay = float(getattr(acc_local, "y", 0.0)) if acc_local is not None else 0.0
                az = float(getattr(acc_local, "z", 0.0)) if acc_local is not None else 0.0

                pw = R @ np.array([px, py, pz]) + p
                vw = R @ np.array([vx, vy, vz])
                aw = R @ np.array([ax, ay, az])

                pos_buf[k_valid] = pw
                vel_buf[k_valid] = vw
                acc_buf[k_valid] = aw
                k_valid += 1

        with self._poses_lock:
            if k_valid > 0:
                self.kinematics_[topic_name] = (
                    pos_buf[:k_valid].copy(),
                    vel_buf[:k_valid].copy(),
                    acc_buf[:k_valid].copy(),
                )
            else:
                z = np.zeros((0, 3), dtype=float)
                self.kinematics_[topic_name] = (z, z.copy(), z.copy())
            self._kin_last_recv_[topic_name] = self.get_clock().now()

    def _get_transform_matrix_to_world(self, frame_id: str, stamp: Any) -> Optional[np.ndarray]:
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
        except Exception:
            last = self._last_tf_warn_time.get(frame_id, 0.0)
            now = time.monotonic()
            if now - last > 2.0:
                self.get_logger().warn(
                    f"TF world <- '{frame_id}' unavailable. Obstacles will stay in '{frame_id}'."
                )
                self._last_tf_warn_time[frame_id] = now
            return None

        tx = ts.transform.translation.x
        ty = ts.transform.translation.y
        tz = ts.transform.translation.z
        qx = ts.transform.rotation.x
        qy = ts.transform.rotation.y
        qz = ts.transform.rotation.z
        qw = ts.transform.rotation.w

        R = self._quat_to_rot(qx, qy, qz, qw)
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]

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

    def getObstacles(self, elapsed: float = 0.0, max_age_sec: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate obstacle kinematic positions, velocities, and accelerations within max_age_sec."""
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

    def switch_to_forward_position_controller_service(self, timeout_sec: float = 10.0) -> None:
        """Call /controller_manager/switch_controller to switch from trajectory controller to forward position controller."""
        client = self.create_client(SwitchController, "/controller_manager/switch_controller")
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError("/controller_manager/switch_controller service not available")

        req = SwitchController.Request()
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
            t0 = time.time()
            while not future.done():
                if time.time() - t0 > timeout_sec:
                    raise TimeoutError("Timeout waiting for switch_controller service response")
                time.sleep(0.01)

        resp = future.result()
        if resp is None:
            raise RuntimeError("switch_controller service call failed (no response)")
        ok = getattr(resp, "ok", True)
        if not ok:
            raise RuntimeError("switch_controller service returned ok=False")
        self.get_logger().info("Controller switch request completed successfully.")

    def shutdown(self) -> None:
        """Shut down background executor and destroy ROS node."""
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:
                pass
            if self._spin_thread and self._spin_thread.is_alive():
                try:
                    self._spin_thread.join(timeout=1.0)
                except Exception:
                    pass
            self._executor = None
        self.destroy_node()
