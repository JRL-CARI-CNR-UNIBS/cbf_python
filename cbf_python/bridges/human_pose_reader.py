"""
Time-stamped human keypoint CSV reader and interpolator with Pinocchio SE3 coordinate frame transforms.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
import numpy as np
import pandas as pd
import pinocchio as pin


class PoseReader:
    """Read a time-stamped keypoint CSV file and provide linearly interpolated 3D coordinates.

    Supports transforming points from world frame to camera frame using an SE3 transform.
    Can auto-differentiate velocities and accelerations via central finite differences if omitted from CSV.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        Tworld_to_cam: Optional[pin.SE3] = None,
        auto_diff_if_missing: bool = False,
    ) -> None:
        self._csv_path = str(csv_path)
        df = pd.read_csv(csv_path)

        if "time" not in df.columns:
            raise ValueError("CSV must have a 'time' column as its first field.")

        self._times: np.ndarray = df["time"].to_numpy(float)
        if len(self._times) < 2:
            raise ValueError("CSV must contain at least two time samples.")

        dt_series = np.diff(self._times)
        pos_dt = dt_series[dt_series > 0]
        self._dt = float(pos_dt.mean()) if pos_dt.size else float(dt_series.mean())
        if not np.isfinite(self._dt) or self._dt <= 0:
            self._dt = 1.0

        def split_name(col: str) -> Tuple[str, str, Optional[str]]:
            suffix = None
            name = col
            if name.endswith("_vel"):
                suffix = "vel"
                name = name[:-4]
            elif name.endswith("_acc"):
                suffix = "acc"
                name = name[:-4]

            if name.endswith("_x"):
                return name[:-2], "x", suffix
            if name.endswith("_y"):
                return name[:-2], "y", suffix
            if name.endswith("_z"):
                return name[:-2], "z", suffix
            return name, "", suffix

        buckets: Dict[Tuple[str, Optional[str]], Dict[str, np.ndarray]] = {}
        for col in df.columns:
            if col == "time":
                continue
            base, axis, suffix = split_name(col)
            if axis not in {"x", "y", "z"}:
                continue
            key = (base, suffix)
            if key not in buckets:
                buckets[key] = {}
            buckets[key][axis] = pd.to_numeric(df[col], errors="coerce").to_numpy(float)

        def assemble_xyz(d: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
            if all(ax in d for ax in ("x", "y", "z")):
                return np.stack([d["x"], d["y"], d["z"]], axis=1)
            return None

        pos_map: Dict[str, np.ndarray] = {}
        vel_map: Dict[str, np.ndarray] = {}
        acc_map: Dict[str, np.ndarray] = {}

        for (base, suffix), axes in buckets.items():
            if suffix is None:
                arr = assemble_xyz(axes)
                if arr is not None:
                    pos_map[base] = arr

        for (base, suffix), axes in buckets.items():
            if base in pos_map:
                arr = assemble_xyz(axes)
                if arr is not None:
                    if suffix == "vel":
                        vel_map[base] = arr
                    elif suffix == "acc":
                        acc_map[base] = arr

        if not pos_map:
            raise ValueError("No (x,y,z) position triplets found in CSV. Expected columns like keypoint1_x,y,z.")

        self._kp_names: List[str] = sorted(pos_map.keys(), key=lambda s: s.lower())
        K = len(self._kp_names)
        N = len(self._times)

        self._pos_world = np.empty((N, K, 3), dtype=float)
        self._vel_world = None
        self._acc_world = None

        for j, name in enumerate(self._kp_names):
            self._pos_world[:, j, :] = pos_map[name]

        if vel_map or auto_diff_if_missing:
            self._vel_world = np.empty((N, K, 3), dtype=float)
            for j, name in enumerate(self._kp_names):
                if name in vel_map:
                    self._vel_world[:, j, :] = vel_map[name]
                elif auto_diff_if_missing:
                    self._vel_world[:, j, :] = np.gradient(self._pos_world[:, j, :], self._dt, axis=0)
                else:
                    self._vel_world[:, j, :] = np.nan

        if acc_map or auto_diff_if_missing:
            self._acc_world = np.empty((N, K, 3), dtype=float)
            for j, name in enumerate(self._kp_names):
                if name in acc_map:
                    self._acc_world[:, j, :] = acc_map[name]
                elif auto_diff_if_missing:
                    if self._vel_world is not None and np.isfinite(self._vel_world).any():
                        self._acc_world[:, j, :] = np.gradient(self._vel_world[:, j, :], self._dt, axis=0)
                    else:
                        v = np.gradient(self._pos_world[:, j, :], self._dt, axis=0)
                        self._acc_world[:, j, :] = np.gradient(v, self._dt, axis=0)
                else:
                    self._acc_world[:, j, :] = np.nan

        self.n_keypoints = K
        self._total_time = float(self._times[-1] - self._times[0])

        self._Tworld_to_cam: pin.SE3 = (
            Tworld_to_cam if Tworld_to_cam is not None else pin.SE3.Identity()
        )

        # Pre-transform coordinates to camera frame
        R = self._Tworld_to_cam.rotation
        t = self._Tworld_to_cam.translation
        self._pos_cam = self._pos_world @ R.T + t
        self._vel_cam = self._vel_world @ R.T if self._vel_world is not None else None
        self._acc_cam = self._acc_world @ R.T if self._acc_world is not None else None

    def getTotalTime(self) -> float:
        """Return total duration of the recorded trajectory."""
        return self._total_time

    def getHumanPose(
        self, t: float, slowdown_factor: float = 1.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Return interpolated keypoint positions, velocities, and accelerations at time t in camera frame.

        Parameters
        ----------
        t : float
            Evaluation time.
        slowdown_factor : float
            Speed factor applied to derivatives (positive = forward, negative = backward).

        Returns
        -------
        pos : np.ndarray
            Keypoint positions of shape (K, 3).
        vel : np.ndarray or None
            Keypoint velocities of shape (K, 3).
        acc : np.ndarray or None
            Keypoint accelerations of shape (K, 3).
        """
        times = self._times
        pC = self._pos_cam
        vC = self._vel_cam
        aC = self._acc_cam

        t0 = times[0]
        tN = times[-1]
        duration = tN - t0

        if duration <= 0.0:
            idx_left = idx_right = 0
            alpha = 0.0
        else:
            t_query = t0 + ((t - t0) % duration)
            if t_query <= times[0]:
                idx_left = idx_right = 0
                alpha = 0.0
            elif t_query >= times[-1]:
                idx_left = idx_right = len(times) - 1
                alpha = 0.0
            else:
                idx_right = int(np.searchsorted(times, t_query, side="right"))
                idx_left = idx_right - 1
                tL = times[idx_left]
                tR = times[idx_right]
                alpha = (t_query - tL) / (tR - tL)

        if idx_left == idx_right:
            pos = pC[idx_left]
            vel = slowdown_factor * vC[idx_left] if vC is not None else None
            acc = (
                np.sign(slowdown_factor) * (slowdown_factor ** 2) * aC[idx_left]
                if aC is not None else None
            )
        else:
            wL = 1.0 - alpha
            wR = alpha
            pos = wL * pC[idx_left] + wR * pC[idx_right]
            vel = (
                slowdown_factor * (wL * vC[idx_left] + wR * vC[idx_right])
                if vC is not None else None
            )
            acc = (
                np.sign(slowdown_factor) * (slowdown_factor ** 2) * (wL * aC[idx_left] + wR * aC[idx_right])
                if aC is not None else None
            )

        return pos, vel, acc
