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
        slowdown_factor: float = 0.4,
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
