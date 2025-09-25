import numpy as np
import pandas as pd
import pinocchio as pin       # pip install pin
from pathlib import Path
from typing import List, Optional

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
    """

    def __init__(
        self,
        csv_path: str | Path,
        Tworld_to_cam: Optional[pin.SE3] = None,
    ) -> None:
        """
        Parameters
        ----------
        csv_path : str | pathlib.Path
            Path to the CSV file on disk.
        Tworld_to_cam : pinocchio.SE3, optional
            Rigid transform that maps world-frame points into the camera frame.
            If None (default) the identity transform is used.
        """
        df = pd.read_csv(csv_path)

        # -------- basic validation -------------------------------------------
        if "time" not in df.columns:
            raise ValueError("CSV must have a 'time' column as its first field.")

        n_coord_cols = df.shape[1] - 1          # exclude time column
        if n_coord_cols % 3:
            raise ValueError(
                "Number of coordinate columns must be divisible by 3 (x, y, z per key-point)."
            )

        # -------- reshape data -----------------------------------------------
        self._times: np.ndarray = df["time"].to_numpy(float)          # (N,)
        n_keypoints = n_coord_cols // 3

        kp_matrix = df.drop(columns=["time"]).to_numpy(float)         # (N, n_keypoints*3)
        self._keypoints = kp_matrix.reshape(len(df), n_keypoints, 3)  # (N, K, 3)
        self.n_keypoints = n_keypoints
        self._total_time = self._times[-1]

        # -------- store transform --------------------------------------------
        self._Tworld_to_cam: pin.SE3 = (
            Tworld_to_cam if Tworld_to_cam is not None else pin.SE3.Identity()
        )

    # -------------------------------------------------------------------------
    def getTotalTime(self) -> float:
        """Total recording duration in the same units as the CSV."""
        return self._total_time

    # -------------------------------------------------------------------------
    def getHumanPose(self, t: float) -> List[np.ndarray]:
        """
        Linearly interpolate the pose at time *t* and convert it
        to the camera frame (Tworld_to_cam * p).

        Parameters
        ----------
        t : float
            Query time stamp.

        Returns
        -------
        List[np.ndarray]
            One 3-D NumPy array (x, y, z) per key-point **in camera coordinates**.
        """
        times = self._times
        kps   = self._keypoints

        # wrap time for looping playback
        t = t % self._total_time

        # edge cases ----------------------------------------------------------
        if t <= times[0]:
            pose_world = kps[0]
        elif t >= times[-1]:
            pose_world = kps[-1]
        else:
            # times[left] ≤ t < times[right]
            idx_right = np.searchsorted(times, t, side="right")
            idx_left  = idx_right - 1

            t0, t1 = times[idx_left], times[idx_right]
            alpha  = (t - t0) / (t1 - t0)          # ∈ [0,1)

            pose_world = (1.0 - alpha) * kps[idx_left] + alpha * kps[idx_right]

        # -------- transform to camera frame ----------------------------------
        T = self._Tworld_to_cam
        pose_cam = [T.act(p) for p in pose_world]   # each result is an np.ndarray(3,)

        return pose_cam
