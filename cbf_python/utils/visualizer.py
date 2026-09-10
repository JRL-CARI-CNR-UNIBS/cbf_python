"""Visualization tools for 3D simulation, metric diagnostics, and parameter tuning."""

import threading
import time
from typing import Any, List, Optional

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import meshcat.geometry as g
import meshcat.transformations as tf
import meshcat_shapes
import numpy as np
from pinocchio.visualize import MeshcatVisualizer
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class VisualizationDaemon:
    """Runs a Meshcat refresher in an independent daemon thread.

    The control loop calls push_state with the latest data; the background thread
    renders the most recent snapshot without stalling the real-time servo loop.
    """

    def __init__(self, viz: MeshcatVisualizer, refresh_hz: float = 60.0) -> None:
        self.viz = viz
        self.refresh_hz = refresh_hz
        self._lock = threading.Lock()

        self._q: Optional[np.ndarray] = None
        self._Tgoal: np.ndarray = np.eye(4)
        self._obstacles: List[np.ndarray] = []
        self._obstacle_velocities: List[np.ndarray] = []
        self._viz_string: str = ""
        self._path: List[np.ndarray] = []

        self._hud = self.viz.viewer["/overlay/speed_text"]
        self._hud.set_transform(tf.translation_matrix([1.0, -0.5, 1.4]))
        self._pathview = self.viz.viewer["/path"]

        self._created_obstacles: set[str] = set()
        self._created_goal: bool = False

        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def push_state(
        self,
        q: np.ndarray,
        Tgoal: Any,
        obstacles: List[np.ndarray],
        obstacle_velocities: Optional[List[np.ndarray]] = None,
        viz_string: str = "",
    ) -> None:
        """Update latest simulation state thread-safely in O(1) time."""
        if not self._lock.locked():
            with self._lock:
                self._q = q.copy()
                if hasattr(Tgoal, "homogeneous"):
                    self._Tgoal = Tgoal.homogeneous.copy()
                else:
                    self._Tgoal = np.asarray(Tgoal).copy()
                self._obstacles = [p.copy() for p in obstacles]
                if obstacle_velocities is not None:
                    self._obstacle_velocities = [v.copy() for v in obstacle_velocities]
                else:
                    self._obstacle_velocities = []
                self._viz_string = str(viz_string)

    def publish_path(self, pts: np.ndarray) -> None:
        """Publish a continuous 3D poly-line connecting path waypoints."""
        with self._lock:
            self._path = pts.copy()

    def _flush(self) -> None:
        if not self._lock.acquire(blocking=False):
            return

        try:
            q_copy = self._q.copy() if self._q is not None else None
            Tgoal_copy = self._Tgoal.copy() if self._Tgoal is not None else None
            obstacles_copy = [p.copy() for p in self._obstacles] if self._obstacles else None
            obs_vel_copy = [v.copy() for v in self._obstacle_velocities] if self._obstacle_velocities else None
            viz_string_copy = str(self._viz_string)

            path_src = self._path
            if path_src is None or len(path_src) == 0:
                path_copy = None
            elif isinstance(path_src, np.ndarray):
                path_copy = path_src.copy()
                self._path = path_src[:0].copy()
            else:
                path_copy = list(path_src)
                self._path = []
        finally:
            self._lock.release()

        if q_copy is not None:
            self.viz.display(q_copy)

        if Tgoal_copy is not None:
            if not self._created_goal:
                side = 0.2
                self.viz.viewer["goal"].set_object(
                    g.Box([side, side, side / 10]), g.MeshLambertMaterial(color=0x00FF00)
                )
                self._created_goal = True
            self.viz.viewer["goal"].set_transform(Tgoal_copy)

        if obstacles_copy is not None:
            for i, pos in enumerate(obstacles_copy):
                node_name = f"obstacle_{i}"
                if node_name not in self._created_obstacles:
                    color = 0x000000 if i == 7 else 0xFF0000
                    self.viz.viewer[node_name].set_object(
                        g.Sphere(0.1), g.MeshLambertMaterial(color=color)
                    )
                    self._created_obstacles.add(node_name)
                self.viz.viewer[node_name].set_transform(tf.translation_matrix(pos))

        if viz_string_copy:
            meshcat_shapes.textarea(self._hud, viz_string_copy, width=1.5, height=1.0, font_size=80)

        if path_copy is not None and len(path_copy) > 0:
            vertices = np.asarray(path_copy, dtype=float).T
            line_geom = g.LineLoop(g.PointsGeometry(vertices), g.LineBasicMaterial(color=0xFF0000))
            self._pathview.set_object(line_geom)

    def _thread_main(self) -> None:
        dt = 1.0 / self.refresh_hz
        while True:
            self._flush()
            time.sleep(dt)


class StochasticCBFVisualizer:
    """Collects stochastic safety observations and computes mean vector and covariance matrix."""

    def __init__(self, n: int = 50) -> None:
        self.n = n
        self.h_vec: List[float] = []
        self.d_vec: List[float] = []
        self.v_vec: List[float] = []
        self.time_vec: List[float] = []
        self.cov_matrix: Optional[np.ndarray] = None
        self.h_mean: Optional[float] = None
        self.d_mean: Optional[float] = None
        self.v_mean: Optional[float] = None

    def update_vectors(self, h: float, d: float, v_rel: float, t: float) -> None:
        self.h_vec.append(h)
        self.d_vec.append(d)
        self.v_vec.append(v_rel)
        self.time_vec.append(t)

    def compute_mean_cov(self, print_val: bool = False) -> None:
        if not self.h_vec:
            return
        data_matrix = np.vstack((self.h_vec, self.d_vec, self.v_vec))
        self.cov_matrix = np.cov(data_matrix)
        self.h_mean = float(np.mean(self.h_vec))
        self.d_mean = float(np.mean(self.d_vec))
        self.v_mean = float(np.mean(self.v_vec))

        if print_val:
            print("--- Data Means ---")
            print(f"h: {self.h_mean:.4f}, d: {self.d_mean:.4f}, v: {self.v_mean:.4f}")
            print("--- Covariance Matrix ---")
            print(np.round(self.cov_matrix, 4))


def make_summary_figure(
    computation_times: Any,
    h_log: Any,
    trj_error_log: Any,
    scaling_log: Any,
    nbins: int = 100,
    height: int = 900,
    show: bool = True,
) -> go.Figure:
    """Build and optionally display an interactive 4-panel diagnostic Plotly figure."""
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Computation Time Distribution",
            "Safety Margin h (evolution)",
            "Trajectory Error (evolution)",
            "Time-Scaling Factor (evolution)",
        ),
        row_heights=[0.25, 0.25, 0.25, 0.25],
        vertical_spacing=0.12,
    )

    fig.add_trace(go.Histogram(x=computation_times, name="Total", opacity=0.5, nbinsx=nbins), row=1, col=1)
    fig.add_trace(go.Scatter(y=h_log, mode="lines+markers", name="h"), row=2, col=1)
    fig.add_trace(go.Scatter(y=trj_error_log, mode="lines+markers", name="Trajectory error"), row=3, col=1)
    fig.add_trace(go.Scatter(y=scaling_log, mode="lines+markers", name="Scaling factor"), row=4, col=1)

    fig.update_layout(
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=60, r=20, t=80, b=60),
    )

    fig.update_xaxes(title_text="Computation time [s]", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_yaxes(title_text="h", row=2, col=1)
    fig.update_xaxes(title_text="Iteration", row=3, col=1)
    fig.update_yaxes(title_text="Trajectory error [rad]", row=3, col=1)
    fig.update_xaxes(title_text="Iteration", row=4, col=1)
    fig.update_yaxes(title_text="Scaling", row=4, col=1)

    if show:
        fig.show()
    return fig


def plot_lambdas(
    t_list: List[float],
    gamma_list: List[float],
    lambda_pos_list: List[float],
    lambda_vel_list: List[float],
    lambda_acc_list: List[float],
    lambda_scaling_list: List[float],
) -> None:
    """Plot gamma and lambda multipliers over 5 synchronized subplots."""
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(t_list, gamma_list, label=r"$\gamma$", color="purple", linewidth=2)
    axs[0].set_ylabel("Gamma")
    axs[0].legend(loc="best")
    axs[0].grid(True, linestyle=":", alpha=0.7)
    axs[0].set_title("Evolution of Gamma and Lambda Multipliers")

    axs[1].plot(t_list, lambda_pos_list, label=r"$\lambda_{pos}$", color="blue", linewidth=2)
    axs[1].set_ylabel("Pos")
    axs[1].legend(loc="best")
    axs[1].grid(True, linestyle=":", alpha=0.7)

    axs[2].plot(t_list, lambda_vel_list, label=r"$\lambda_{vel}$", color="orange", linewidth=2)
    axs[2].set_ylabel("Vel")
    axs[2].legend(loc="best")
    axs[2].grid(True, linestyle=":", alpha=0.7)

    axs[3].plot(t_list, lambda_acc_list, label=r"$\lambda_{acc}$", color="green", linewidth=2)
    axs[3].set_ylabel("Acc")
    axs[3].legend(loc="best")
    axs[3].grid(True, linestyle=":", alpha=0.7)

    axs[4].plot(t_list, lambda_scaling_list, label=r"$\lambda_{scaling}$", color="red", linewidth=2)
    axs[4].set_ylabel("Scaling")
    axs[4].legend(loc="best")
    axs[4].grid(True, linestyle=":", alpha=0.7)
    axs[4].set_xlabel("Time (t)", fontsize=12)

    plt.tight_layout()
    plt.show()


def calculate_polynomial_lambda(
    eta: np.ndarray, r: float, s: float, t: float, l_inf: float, l_sup: float
) -> np.ndarray:
    """Polynomial transition function for normalized parameter eta in [0, 1]."""
    return l_inf + (l_sup - l_inf) * (r * (eta ** s) + (1.0 - r) * (eta ** t))


def launch_interactive_polynomial_viewer() -> None:
    """Launch interactive Matplotlib window with sliders to visualize polynomial adaptation curves."""
    init_r, init_s, init_t = 1.0, 2.0, 4.0
    init_l_inf, init_l_sup = 1.0, 5.0
    eta = np.linspace(0, 1, 500)

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(left=0.1, bottom=0.4)

    [line] = ax.plot(
        eta,
        calculate_polynomial_lambda(eta, init_r, init_s, init_t, init_l_inf, init_l_sup),
        linewidth=2,
        color="blue",
        label=r"$\lambda(\eta)$",
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Non-negativity constraint")
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 10)
    ax.set_xlabel(r"$\eta$ (Normalized Transition)", fontsize=12)
    ax.set_ylabel(r"$\lambda_i$", fontsize=12)
    ax.set_title("Interactive Weight Adaptation Rule", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="upper left")

    axcolor = "lightgoldenrodyellow"
    ax_r = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_s = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
    ax_t = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_linf = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_lsup = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

    slider_r = Slider(ax_r, "r (Weight)", -5.0, 5.0, valinit=init_r, valstep=0.1)
    slider_s = Slider(ax_s, "s (Exp 1)", 0.1, 5.0, valinit=init_s, valstep=0.1)
    slider_t = Slider(ax_t, "t (Exp 2)", 0.1, 10.0, valinit=init_t, valstep=0.1)
    slider_linf = Slider(ax_linf, r"$\lambda_{inf}$", 0.0, 10.0, valinit=init_l_inf, valstep=0.5)
    slider_lsup = Slider(ax_lsup, r"$\lambda_{sup}$", 0.0, 10.0, valinit=init_l_sup, valstep=0.5)

    def update(_: Any) -> None:
        r = slider_r.val
        s = slider_s.val
        t = slider_t.val
        l_inf = slider_linf.val
        l_sup = slider_lsup.val

        if t <= s:
            slider_t.set_val(s + 0.1)
            t = slider_t.val

        new_y = calculate_polynomial_lambda(eta, r, s, t, l_inf, l_sup)
        line.set_ydata(new_y)
        min_y = min(-2.0, float(np.min(new_y)) - 1.0)
        max_y = max(10.0, float(np.max(new_y)) + 1.0)
        ax.set_ylim(min_y, max_y)
        fig.canvas.draw_idle()

    slider_r.on_changed(update)
    slider_s.on_changed(update)
    slider_t.on_changed(update)
    slider_linf.on_changed(update)
    slider_lsup.on_changed(update)

    plt.show()
