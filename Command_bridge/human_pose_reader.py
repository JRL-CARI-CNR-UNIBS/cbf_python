import numpy as np
import pandas as pd
import pinocchio as pin       # pip install pin
from pathlib import Path
from typing import List, Optional, Dict, Tuple

class PoseReader:
    """
    Read a time-stamped key-point CSV and return linearly-interpolated poses.
    Optionally convert them from *world* to *camera* coordinates with a
    Pinocchio SE3 transform.

    CSV layout (header required):
        time,
        keypoint1_x,keypoint1_y,keypoint1_z,
        keypoint2_x,keypoint2_y,keypoint2_z,
        ...
    Optionally (if present) velocity/acceleration columns are recognized when
    suffixed with `_vel` / `_acc`, e.g.:
        keypoint1_x_vel,keypoint1_y_vel,keypoint1_z_vel,
        keypoint1_x_acc,keypoint1_y_acc,keypoint1_z_acc
    """

    def __init__(
        self,
        csv_path: str | Path,
        Tworld_to_cam: Optional[pin.SE3] = None,
        auto_diff_if_missing: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        csv_path : str | pathlib.Path
            Path to the CSV file on disk.
        Tworld_to_cam : pinocchio.SE3, optional
            Rigid transform that maps world-frame points into the camera frame.
            If None (default) the identity transform is used.
        auto_diff_if_missing : bool
            If True and velocity/acceleration columns are absent, compute them
            via finite differences from positions (uniform dt inferred from time).
        """
        df = pd.read_csv(csv_path)

        # -------- basic validation -------------------------------------------
        if "time" not in df.columns:
            raise ValueError("CSV must have a 'time' column as its first field.")

        # Parse time and dt

        self._times: np.ndarray = df["time"].to_numpy(float)          # (N,)
        if len(self._times) < 2:
            raise ValueError("CSV must contain at least two time samples.")
        # robust average positive dt
        dt_series = np.diff(self._times)
        pos_dt = dt_series[dt_series > 0]
        self._dt = float(pos_dt.mean()) if pos_dt.size else float(dt_series.mean())
        if not np.isfinite(self._dt) or self._dt <= 0:
            self._dt = 1.0

        # -------- identify keypoints & columns -------------------------------
        # We accept names like: keypoint1_x, keypoint1_y, keypoint1_z
        # Optional extras: *_vel, *_acc
        def split_name(col: str) -> Tuple[str, str, Optional[str]]:
            """
            Returns (base_name, axis, suffix)
            base_name: e.g. 'keypoint1'
            axis: one of {'x','y','z'} if matches; else ''
            suffix: None | 'vel' | 'acc'
            """
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

        # Build maps: {keypoint: {pos|vel|acc: (N,3) array}}
        pos_map: Dict[str, np.ndarray] = {}
        vel_map: Dict[str, np.ndarray] = {}
        acc_map: Dict[str, np.ndarray] = {}

        # First collect all columns by (keypoint, suffix, axis)
        buckets: Dict[Tuple[str, Optional[str]], Dict[str, np.ndarray]] = {}

        for col in df.columns:
            if col == "time":
                continue
            base, axis, suffix = split_name(col)
            if axis not in {"x", "y", "z"}:
                # ignore unrelated columns
                continue
            key = (base, suffix)  # suffix None|'vel'|'acc'
            if key not in buckets:
                buckets[key] = {}
            buckets[key][axis] = pd.to_numeric(df[col], errors="coerce").to_numpy(float)

        # Now assemble xyz triplets
        def assemble_xyz(d: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
            if all(ax in d for ax in ("x", "y", "z")):
                return np.stack([d["x"], d["y"], d["z"]], axis=1)  # (N,3)
            return None

        # pass 1: positions (suffix=None)
        for (base, suffix), axes in buckets.items():
            if suffix is None:
                arr = assemble_xyz(axes)
                if arr is not None:
                    pos_map[base] = arr

        # pass 2: velocities, accelerations (if provided)
        for (base, suffix), axes in buckets.items():
            if base in pos_map:
                arr = assemble_xyz(axes)
                if arr is None:
                    continue
                if suffix == "vel":
                    vel_map[base] = arr
                elif suffix == "acc":
                    acc_map[base] = arr

        if not pos_map:
            raise ValueError("No (x,y,z) position triplets found. Expect columns like 'keypoint1_x,y,z'.")

        # Ensure consistent keypoint order
        self._kp_names: List[str] = sorted(pos_map.keys(), key=lambda s: s.lower())
        K = len(self._kp_names)
        N = len(self._times)

        # Create (N,K,3) tensors
        self._pos_world = np.empty((N, K, 3), dtype=float)
        self._vel_world = None
        self._acc_world = None

        for j, name in enumerate(self._kp_names):
            self._pos_world[:, j, :] = pos_map[name]

        # Optionally fill in vel/acc
        if vel_map or auto_diff_if_missing:
            self._vel_world = np.empty((N, K, 3), dtype=float)
            for j, name in enumerate(self._kp_names):
                if name in vel_map:
                    self._vel_world[:, j, :] = vel_map[name]
                elif auto_diff_if_missing:
                    # central differences with edge handling
                    self._vel_world[:, j, :] = np.gradient(self._pos_world[:, j, :], self._dt, axis=0)
                else:
                    # if not auto-diff, set NaNs where missing
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
                        # direct second derivative from positions
                        v = np.gradient(self._pos_world[:, j, :], self._dt, axis=0)
                        self._acc_world[:, j, :] = np.gradient(v, self._dt, axis=0)
                else:
                    self._acc_world[:, j, :] = np.nan

        self.n_keypoints = K
        self._total_time = float(self._times[-1] - self._times[0])

        # -------- store transform --------------------------------------------
        self._Tworld_to_cam: pin.SE3 = (
            Tworld_to_cam if Tworld_to_cam is not None else pin.SE3.Identity()
        )

        # -------- pre-transform WORLD -> CAMERA (ONE-TIME COST) -------------
        T = self._Tworld_to_cam
        R = T.rotation  # (3,3)
        t = T.translation  # (3,)

        # Positions: (N,K,3)
        self._pos_cam = self._pos_world @ R.T + t

        # Velocities (rotation only)
        if self._vel_world is not None:
            self._vel_cam = self._vel_world @ R.T
        else:
            self._vel_cam = None

        # Accelerations (rotation only)
        if self._acc_world is not None:
            self._acc_cam = self._acc_world @ R.T
        else:
            self._acc_cam = None

    # -------------------------------------------------------------------------
    def getTotalTime(self) -> float:
        """Total recording duration in the same units as the CSV."""
        return self._total_time

    def getHumanPose(self, t: float, slowdown_factor: float):
        """
        Fast, allocation-free pose lookup using pre-transformed camera-frame data.

        Returns
        -------
        pos : np.ndarray, shape (K,3)
        vel : np.ndarray or None, shape (K,3)
        acc : np.ndarray or None, shape (K,3)
        """

        times = self._times
        pC = self._pos_cam
        vC = self._vel_cam
        aC = self._acc_cam

        # ---- wrap time into recorded interval ---------------------------------
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

        # ---- interpolate -------------------------------------------------------
        if idx_left == idx_right:
            pos = pC[idx_left]

            vel = None
            if vC is not None:
                vel = slowdown_factor * vC[idx_left]

            acc = None
            if aC is not None:
                acc = (
                        np.sign(slowdown_factor)
                        * slowdown_factor ** 2
                        * aC[idx_left]
                )

        else:
            wL = 1.0 - alpha
            wR = alpha

            pos = wL * pC[idx_left] + wR * pC[idx_right]

            vel = None
            if vC is not None:
                vel = slowdown_factor * (
                        wL * vC[idx_left] + wR * vC[idx_right]
                )

            acc = None
            if aC is not None:
                acc = (
                        np.sign(slowdown_factor)
                        * slowdown_factor ** 2
                        * (wL * aC[idx_left] + wR * aC[idx_right])
                )

        return pos, vel, acc


