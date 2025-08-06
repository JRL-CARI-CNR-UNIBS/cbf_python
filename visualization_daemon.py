
import threading
import time
from typing import List

import meshcat.geometry as g
import meshcat.transformations as tf
import meshcat_shapes
import numpy as np
from pinocchio.visualize import MeshcatVisualizer

import math

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

        # ---------- state updated by the control loop ----------
        self._q: np.ndarray | None = None     # robot configuration
        self._Tgoal = np.eye(4)               # 4×4 goal pose
        self._obstacles: list[np.ndarray] = []  # list of 3-vecs
        self._viz_string = ""                         # string

        # single HUD label reused every frame
        self._hud = viz.viewer["/overlay/speed_text"]

        self._pathview = viz= viz.viewer["/path"]

        self._hud.set_transform(
            tf.translation_matrix([0.0, -0.5, 0.0])
        )

        # launch the thread
        self._thread = threading.Thread(target=self._thread_main,
                                        daemon=True)


        self._path = []
        self._thread.start()



    # ------------------------------------------------------------------ API -
    def push_state(self,
                   q: np.ndarray,
                   Tgoal: np.ndarray,
                   obstacles: list[np.ndarray],
                   viz_string: str = "") -> None:
        """Copy the latest simulation state (O(1) in control loop)."""
        if not self._lock.locked():
            with self._lock:
                self._q = q.copy()
                self._Tgoal = Tgoal.copy().homogeneous
                self._obstacles = [p.copy() for p in obstacles]
                self._viz_string = str(viz_string)

    # ----------------------------------------------------------- internals --
    def _flush(self) -> None:
        """Push the stored state to Meshcat. Skips frame if busy."""
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._q is not None:
                self.viz.display(self._q)

            self.viz.viewer["goal"].set_transform(self._Tgoal)

            for i, pos in enumerate(self._obstacles):
                self.viz.viewer[f"obstacle_{i}"].set_transform(
                    tf.translation_matrix(pos)
                )

            meshcat_shapes.textarea(
                self._hud,
                f"{self._viz_string}",
                width=1.5,
                height=1.0,
                font_size=80,
            )
            if len(self._path)>0:
                print(f"path={self._path}")

                line_geom = g.LineLoop(
                    g.PointsGeometry(np.asarray(self._path, dtype=float).T),
                    g.LineBasicMaterial(color=0xff0000),
                )
                #v["line_loop_with_material"].set_object(
                #    g.LineLoop(g.PointsGeometry(vertices), g.LineBasicMaterial(color=0xff0000)))
                self._pathview.set_object(line_geom)

                self._path = []
        finally:
            self._lock.release()

    def _thread_main(self) -> None:
        dt = 1.0 / self.refresh_hz
        while True:
            self._flush()
            time.sleep(dt)


    def publishPath(self, pts):
        """
        Publish a poly-line (polygonal chain) that connects the stored way-points.

        Parameters
        ----------
        viz_viewer : meshcat.Visualizer
            The MeshCat viewer into which the path is drawn (e.g. ``viz.viewer``).
        name : str, default "path"
            Sub-node name under the viewer where the object is stored.
        color : int, default 0x0080ff
            RGB hex colour for the line.
        linewidth : float, default 4
            Line width in pixels.
        """
        # MeshCat line object: one segment per successive pair of points

        with self._lock:
            self._path = pts.copy()
