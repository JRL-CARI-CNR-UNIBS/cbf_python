#!/usr/bin/env python3
from __future__ import annotations

import time
from typing import Iterable, Tuple, Optional, Callable

import numpy as np
import pinocchio as pin

from base_command_bridge_abc import BaseCommandBridgeABC
from human_pose_reader import PoseReader  # provides PoseReader(csv, Tworld_to_cam, ...)  :contentReference[oaicite:1]{index=1}


class FakeCommandBridge(BaseCommandBridgeABC):
    """
    Topic-free, ROS-free bridge that simulates human poses from a CSV using PoseReader.
    Implements the BaseCommandBridgeABC interface:

      - _do_publish: stores the last command (and optional callback)
      - getObstacles(max_age_sec): returns (pos[K,3], vel[K,3], acc[K,3]) at the current simulated time

    Time evolution:
      - Let T = human_trj_time.
      - We advance with a slowdown_factor (default 0.4).
      - t_raw = slowdown_factor * (now - t0).
      - t_human = triangular-wave mirror over [0, 2T]:
            if 0 <= (t_raw mod 2T) <= T  -> forward
            else                          -> backward with mirrored time and NEGATED slowdown
    """

    def __init__(
        self,
        ordered_joint_names: Iterable[str],
        *,
        threshold: float = 0.05,
        csv_path: str = "a01_s10_e02_skeleton3D_with_savgol_vel_acc.csv",
        Tworld_to_cam: Optional[pin.SE3] = None,
        slowdown_factor: float = 0.4,
        on_publish: Optional[Callable[[np.ndarray], None]] = None,
        auto_diff_if_missing: bool = False,
        t0=time.monotonic(),
    ) -> None:
        super().__init__(ordered_joint_names,threshold=threshold)

        # --- Build default Tworld->cam if none provided (INITI snippet) ---
        if Tworld_to_cam is None:
            R = pin.utils.rotate('z', 1.9) @ pin.utils.rotate('x', 1.57)  # ~90° about X, 1.9 rad about Z
            Tworld_to_cam = pin.SE3(R, np.array([-1.85, -0.9, 0.9]))

        # --- Pose reader & trajectory duration ---
        self._reader = PoseReader(csv_path, Tworld_to_cam, auto_diff_if_missing=auto_diff_if_missing)  # :contentReference[oaicite:2]{index=2}
        self._human_T = float(self._reader.getTotalTime())  # recording duration  :contentReference[oaicite:3]{index=3}

        # --- Simulation knobs ---
        self._slowdown = float(slowdown_factor)
        self._t0 = t0
        # print("T0: ", self._t0)
        self._on_publish = on_publish
        self.last_command: Optional[np.ndarray] = None  # for inspection/tests
        self.actual_joint_positions_ = np.array([90.0, -140.0, 140.0, -90.0, 90.0, 0.0]) * np.pi / 180.0
        self.last_command = self.actual_joint_positions_
        self.actual_joint_velocities_= np.array([0.0]*6)
        self.actual_joint_accelerations_ = np.array([0.0] * 6)


    # --------------------- ABC: command publish ---------------------
    def _do_publish(self, q: np.ndarray) -> None:
        self.last_command = q.copy()
        if self._on_publish is not None:
            try:
                self._on_publish(q.copy())
            except Exception:
                pass  # publishing callback is best-effort here

    # --------------------- ABC: obstacles provider ------------------
    def getObstacles(self, elapsed = time.monotonic(), max_age_sec: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (pos[K,3], vel[K,3], acc[K,3]) for the current simulated time.

        The CSV may contain any number of keypoints K; arrays are shaped (K,3).
        """
        # current raw time scaled by slowdown
        elapsed = elapsed - self._t0
        # ("ELAPSED: ", elprintapsed)
        t_raw = self._slowdown * elapsed

        T = self._human_T
        # wrap into [0, 2T)
        tw = t_raw % (2.0 * T) if T > 0 else 0.0

        # mirror (triangular wave) + sign flip of slowdown on the backward leg
        if tw <= T:
            t_human = tw
            eff_slowdown = self._slowdown
        else:
            t_human = 2.0 * T - tw
            eff_slowdown = -self._slowdown

        # sample pose from reader (camera frame): lists of length K, each (3,)
        pos_list, vel_list, acc_list = self._reader.getHumanPose(t_human, eff_slowdown)  # :contentReference[oaicite:4]{index=4}

        # convert to (K,3) arrays
        pos = np.asarray(pos_list, dtype=float).reshape(-1, 3)
        vel = np.asarray(vel_list, dtype=float).reshape(-1, 3)
        acc = np.asarray(acc_list, dtype=float).reshape(-1, 3)
        return pos, vel, acc
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

    # --------------------- Convenience controls ---------------------
    def reset_time(self) -> None:
        """Restart the simulated time origin."""
        self._t0 = time.monotonic()

    def set_slowdown(self, value: float) -> None:
        """Change how fast we traverse the trajectory (sign controls direction)."""
        self._slowdown = float(value)

    def set_world_to_cam(self, T: pin.SE3) -> None:
        """Swap the camera frame on the fly (e.g., if you move the sensor)."""
        # Recreate the reader to apply a different transform without re-parsing CSV
        # (keeps timing/arrays intact by copying internals).
        self._reader = PoseReader(self._reader._csv_path, T)  # relies on PoseReader’s API and cached CSV  :contentReference[oaicite:5]{index=5}
