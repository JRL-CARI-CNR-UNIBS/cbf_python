"""
Simulated, ROS-independent command and obstacle bridge using PoseReader.
"""

from __future__ import annotations
import time
from typing import Iterable, Tuple, Optional, Callable
import numpy as np
import pinocchio as pin

from cbf_python.bridges.base_bridge import BaseCommandBridgeABC
from cbf_python.bridges.human_pose_reader import PoseReader


class FakeCommandBridge(BaseCommandBridgeABC):
    """Topic-free, ROS-independent bridge that simulates human obstacle motion from recorded CSV trajectories.

    Implements the BaseCommandBridgeABC interface:
      - _do_publish: stores the commanded joint state and optionally triggers an on_publish callback.
      - getObstacles: returns (pos[K, 3], vel[K, 3], acc[K, 3]) at simulated elapsed time.
    """

    def __init__(
        self,
        ordered_joint_names: Iterable[str],
        *,
        threshold: float = 0.05,
        csv_path: Optional[str] = None,
        Tworld_to_cam: Optional[pin.SE3] = None,
        slowdown_factor: float = 1.0,
        on_publish: Optional[Callable[[np.ndarray], None]] = None,
        auto_diff_if_missing: bool = False,
        t0: Optional[float] = None,
    ) -> None:
        super().__init__(ordered_joint_names, threshold=threshold)

        from cbf_python.utils.config_loader import resolve_path
        if csv_path is None:
            csv_path = str(resolve_path("skeleton_vectors/skeleton_vectors_23.csv"))
        else:
            csv_path = str(resolve_path(csv_path))

        if Tworld_to_cam is None:
            R = pin.utils.rotate("z", 1.9) @ pin.utils.rotate("x", 1.57)
            Tworld_to_cam = pin.SE3(R, np.array([-1.85, -0.9, 0.9]))

        self._reader = PoseReader(csv_path, Tworld_to_cam, auto_diff_if_missing=auto_diff_if_missing)
        self._human_T = float(self._reader.getTotalTime())

        self._slowdown = float(slowdown_factor)
        self._t0 = time.monotonic() if t0 is None else float(t0)
        self._on_publish = on_publish

        n = len(self.ordered_joint_names_)
        default_pos = np.array([90.0, -140.0, 140.0, -90.0, 90.0, 0.0]) * np.pi / 180.0
        if n == len(default_pos):
            self.actual_joint_positions_ = default_pos.copy()
        else:
            self.actual_joint_positions_ = np.zeros(n, dtype=float)

        self.last_command = self.actual_joint_positions_.copy()
        self.actual_joint_velocities_ = np.zeros(n, dtype=float)
        self.actual_joint_accelerations_ = np.zeros(n, dtype=float)

    def _do_publish(self, q: np.ndarray) -> None:
        self.last_command = q.copy()
        with self._state_lock:
            self.actual_joint_positions_ = q.copy()
        if self._on_publish is not None:
            try:
                self._on_publish(q.copy())
            except Exception:
                pass

    def getObstacles(
        self, elapsed: Optional[float] = None, max_age_sec: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (pos[K, 3], vel[K, 3], acc[K, 3]) for current simulation time."""
        if elapsed is None:
            t_curr = time.monotonic() - self._t0
        else:
            t_curr = elapsed

        t_raw = self._slowdown * t_curr
        T = self._human_T
        tw = (t_raw % (2.0 * T)) if T > 0.0 else 0.0

        if tw <= T:
            t_human = tw
            eff_slowdown = self._slowdown
        else:
            t_human = 2.0 * T - tw
            eff_slowdown = -self._slowdown

        pos_arr, vel_arr, acc_arr = self._reader.getHumanPose(t_human, eff_slowdown)

        pos = np.asarray(pos_arr, dtype=float).reshape(-1, 3)
        vel = np.asarray(vel_arr, dtype=float).reshape(-1, 3) if vel_arr is not None else np.zeros_like(pos)
        acc = np.asarray(acc_arr, dtype=float).reshape(-1, 3) if acc_arr is not None else np.zeros_like(pos)

        return pos, vel, acc

    def reset_time(self) -> None:
        """Reset the internal time origin."""
        self._t0 = time.monotonic()

    def set_slowdown(self, value: float) -> None:
        """Set speed multiplier."""
        self._slowdown = float(value)

    def set_world_to_cam(self, T: pin.SE3) -> None:
        """Update the camera frame placement."""
        self._reader = PoseReader(self._reader._csv_path, T)

    def shutdown(self) -> None:
        """Shut down fake bridge (no-op as it does not spawn background ROS executors)."""
        pass


class MultiRegionGestureSequencer:
    """Manages multi-region predictive human disturbance sequencing across nominal path segments.

    Enables high-density, multi-waypoint collaborative human-robot interactions without
    artificial freezing or time-drift desynchronization. Predictively launches authentic
    recorded motion-capture gestures so their peak intrusions align with the robot's
    arrival at specific sectors (e.g. center workstation, left edge, return boundary).
    """

    def __init__(
        self,
        bridge: FakeCommandBridge,
        gestures: Optional[List[dict]] = None,
        T_total: float = 14.93,
    ) -> None:
        self.bridge = bridge
        self.T_total = float(T_total)

        if not gestures:
            self.gestures = [
                {
                    "name": "Center Workstation Reach",
                    "target_xi": 0.475,
                    "lead_time": 2.30,
                    "csv_start": 0.50,
                    "csv_end": 6.20,
                },
                {
                    "name": "Left Edge Reach",
                    "target_xi": 0.770,
                    "lead_time": 2.00,
                    "csv_start": 7.50,
                    "csv_end": 10.50,
                },
                {
                    "name": "Return Sweep Reach",
                    "target_xi": 0.890,
                    "lead_time": 0.90,
                    "csv_start": 10.80,
                    "csv_end": 13.20,
                },
            ]
        else:
            self.gestures = [dict(g) for g in gestures]

        for g in self.gestures:
            g["target_s"] = float(g.get("target_xi", 0.5)) * self.T_total
            g["lead_time"] = float(g.get("lead_time", 2.0))
            g["csv_start"] = float(g.get("csv_start", 0.0))
            g["csv_end"] = float(g.get("csv_end", 5.0))
            g["duration"] = g["csv_end"] - g["csv_start"]

        self.current_idx = 0
        self.active_gesture: Optional[dict] = None
        self.t_trigger: float = 0.0

    def step(
        self, t: float, trajectory_time: float, curr_scaling: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str]]:
        """Advance the sequencer and return (obs_pos, obs_vel, obs_acc, event_msg)."""
        curr_s = trajectory_time % self.T_total
        v_est = max(0.05, curr_scaling)
        event_msg: Optional[str] = None

        g = self.gestures[self.current_idx]

        if self.active_gesture is None:
            target_s = g["target_s"]
            lead_time = g["lead_time"]

            if curr_s < target_s:
                t_eta = (target_s - curr_s) / v_est
            else:
                t_eta = (target_s + self.T_total - curr_s) / v_est

            if t_eta <= lead_time and (target_s - curr_s) > -0.5:
                self.active_gesture = g
                self.t_trigger = t
                event_msg = (
                    f"[Multi-Region Trigger] Launched {g['name']} at t={t:.2f}s "
                    f"(robot s={curr_s:.2f}s, xi={curr_s/self.T_total:.3f}, ETA={t_eta:.2f}s)"
                )

        if self.active_gesture is not None:
            elapsed_in_g = t - self.t_trigger
            if elapsed_in_g >= self.active_gesture["duration"]:
                self.active_gesture = None
                self.current_idx = (self.current_idx + 1) % len(self.gestures)
                pos, vel, acc = self.bridge.getObstacles(elapsed=0.0)
                vel = np.zeros_like(vel)
                acc = np.zeros_like(acc)
            else:
                csv_t = self.active_gesture["csv_start"] + elapsed_in_g
                pos, vel, acc = self.bridge.getObstacles(elapsed=csv_t)
        else:
            pos, vel, acc = self.bridge.getObstacles(elapsed=0.0)
            vel = np.zeros_like(vel)
            acc = np.zeros_like(acc)

        return pos, vel, acc, event_msg

    def reset(self) -> None:
        """Reset sequencer state."""
        self.current_idx = 0
        self.active_gesture = None
        self.t_trigger = 0.0

