import threading
import time
from typing import List, Optional

import meshcat.geometry as g
import meshcat.transformations as tf
import meshcat_shapes
import numpy as np
from pinocchio.visualize import MeshcatVisualizer


# ---------------------------------------------------------------------------
#  Meshcat background renderer – thread-safe, non-blocking, 60 Hz by default
# ---------------------------------------------------------------------------
class VisualizationDaemon:
    """Runs a Meshcat refresher in its own daemon thread.

    The control loop calls `push_state(…)` whenever it has new data; the
    background thread renders the most recent snapshot, skipping a frame if
    the previous one is still in flight (so the servo loop never stalls).
    """

    def __init__(self,
                 viz: MeshcatVisualizer,
                 refresh_hz: float = 60.0):
        self.viz = viz
        self.refresh_hz = refresh_hz
        self._lock = threading.Lock()

        # ---------- State updated by the control loop ----------
        self._q: Optional[np.ndarray] = None  # Robot configuration
        self._Tgoal: np.ndarray = np.eye(4)  # 4x4 goal pose
        self._obstacles: List[np.ndarray] = []  # List of 3D positions
        self._obstacle_velocities: List[np.ndarray] = []  # List of 3D velocities
        self._viz_string: str = ""  # HUD text
        self._path: List[np.ndarray] = []  # Path waypoints

        # Single HUD label reused every frame
        self._hud = self.viz.viewer["/overlay/speed_text"]
        self._hud.set_transform(tf.translation_matrix([1.0, -0.5, 1.4]))

        # Path viewer node
        self._pathview = self.viz.viewer["/path"]

        # Launch the daemon thread
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ API -
    def push_state(self,
                   q: np.ndarray,
                   Tgoal: np.ndarray,
                   obstacles: List[np.ndarray],
                   obstacle_velocities: Optional[List[np.ndarray]] = None,
                   viz_string: str = "") -> None:
        """Copy the latest simulation state (O(1) in control loop)."""
        if not self._lock.locked():
            with self._lock:
                self._q = q.copy()
                # Convert SE3 to a 4x4 numpy array if needed
                if hasattr(Tgoal, 'homogeneous'):
                    self._Tgoal = Tgoal.homogeneous.copy()
                else:
                    self._Tgoal = Tgoal.copy()
                self._obstacles = [p.copy() for p in obstacles]

                if obstacle_velocities is not None:
                    self._obstacle_velocities = [v.copy() for v in obstacle_velocities]
                else:
                    self._obstacle_velocities = []

                self._viz_string = str(viz_string)

    def publishPath(self, pts: np.ndarray) -> None:
        """
        Publish a poly-line (polygonal chain) that connects the stored waypoints.
        """
        with self._lock:
            self._path = pts.copy()

    # ----------------------------------------------------------- Internals --
    def _flush(self) -> None:
        """Push the stored state to Meshcat. Locks only to copy state, then releases."""
        # Try to take the lock non-blocking; if busy, skip this rendering frame.
        if not self._lock.acquire(blocking=False):
            return

        # ---- Copy-only critical section ----
        try:
            q_copy = self._q.copy() if self._q is not None else None
            Tgoal_copy = self._Tgoal.copy() if self._Tgoal is not None else None

            obstacles_copy = [p.copy() for p in self._obstacles] if self._obstacles else None
            obs_vel_copy = [v.copy() for v in self._obstacle_velocities] if self._obstacle_velocities else None

            viz_string_copy = str(self._viz_string)

            # Copy and clear _path atomically
            path_src = self._path
            if path_src is None or len(path_src) == 0:
                path_copy = None
            elif isinstance(path_src, np.ndarray):
                path_copy = path_src.copy()
                self._path = path_src[:0].copy()  # Reset to empty array maintaining shape
            else:
                path_copy = list(path_src)
                try:
                    path_src.clear()
                except AttributeError:
                    self._path = []
        finally:
            self._lock.release()
        # ---- End critical section ----

        # Now, outside the lock, do the heavier visualization work in Meshcat.
        if q_copy is not None:
            self.viz.display(q_copy)

        if Tgoal_copy is not None:
            self.viz.viewer["goal"].set_transform(Tgoal_copy)

        # Render obstacles and their velocity arrows
        if obstacles_copy is not None:
            for i, pos in enumerate(obstacles_copy):
                # 1. Update the position of the obstacle's base sphere
                self.viz.viewer[f"obstacle_{i}"].set_transform(tf.translation_matrix(pos))

                # 2. Update the velocity arrow (cylinder)
                if obs_vel_copy and i < len(obs_vel_copy):
                    vel = obs_vel_copy[i]
                    v_norm = np.linalg.norm(vel)

                    if v_norm > 1e-4:  # If the obstacle is actively moving
                        v_dir = vel / v_norm

                        # Velocity scaling factor for visualization purposes
                        scale_factor = 7.5
                        length = v_norm * scale_factor

                        # Assuming the obstacle sphere has a radius of 0.1, we start
                        # the cylinder right at the surface of the sphere.
                        # The center of a cylinder in Meshcat is located at half its height.
                        sphere_radius = 0.1
                        center_pos = pos + v_dir * (sphere_radius + length / 2.0)

                        # Build an orthonormal basis to align the cylinder's Y-axis to v_dir
                        y_vec = v_dir
                        if abs(y_vec[2]) < 0.99:
                            temp = np.array([0.0, 0.0, 1.0])
                        else:
                            temp = np.array([1.0, 0.0, 0.0])

                        x_vec = np.cross(y_vec, temp)
                        x_vec /= np.linalg.norm(x_vec)
                        z_vec = np.cross(x_vec, y_vec)

                        # Construct the 4x4 homogeneous transformation matrix
                        T_arrow = np.eye(4)
                        T_arrow[:3, :3] = np.column_stack((x_vec, y_vec, z_vec))
                        T_arrow[:3, 3] = center_pos

                        # Create a thick cylinder for high visibility against the black sphere
                        arrow_geom = g.Cylinder(height=length, radius=0.015)
                        arrow_mat = g.MeshLambertMaterial(color=0x00FFFF)  # Orange-Red color

                        self.viz.viewer[f"obstacle_vel_{i}"].set_object(arrow_geom, arrow_mat)
                        self.viz.viewer[f"obstacle_vel_{i}"].set_transform(T_arrow)
                    else:
                        # If velocity is near zero, hide the arrow by scaling it down to zero
                        self.viz.viewer[f"obstacle_vel_{i}"].set_transform(tf.scale_matrix(0.001))

        # Update HUD text
        if viz_string_copy:
            meshcat_shapes.textarea(
                self._hud,
                viz_string_copy,
                width=1.5,
                height=1.0,
                font_size=80,
            )

        # Render the planned path as a continuous red line
        if path_copy is not None and len(path_copy) > 0:
            # MeshCat expects vertices in a 3xN shape
            vertices = np.asarray(path_copy, dtype=float).T
            line_geom = g.LineLoop(
                g.PointsGeometry(vertices),
                g.LineBasicMaterial(color=0xff0000),
            )
            self._pathview.set_object(line_geom)

    def _thread_main(self) -> None:
        """The main loop of the background rendering thread."""
        dt = 1.0 / self.refresh_hz
        while True:
            self._flush()
            time.sleep(dt)