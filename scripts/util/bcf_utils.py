from typing import Sequence, Dict, Any, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def make_summary_figure(
    computation_times: Sequence[float],
    h_log: Sequence[float],
    trj_error_log: Sequence[float],
    scaling_log: Sequence[float],
    nbins: int = 100,
    height: int = 900,
    show: bool = True,
) -> go.Figure:
    """
    Build and optionally render a 4-panel interactive summary figure using Plotly.

    Panels:
      1) Histogram of total computation time per cycle [s]
      2) Safety barrier margin h evolution
      3) Trajectory tracking error evolution [rad]
      4) Velocity time-scaling factor evolution s_dot

    Parameters:
        computation_times : Sequence[float] - Control loop runtimes [s].
        h_log             : Sequence[float] - Minimum barrier function values over iterations.
        trj_error_log     : Sequence[float] - Joint tracking errors [rad].
        scaling_log       : Sequence[float] - Time-scaling factor Dtrajectory_time.
        nbins             : int - Number of histogram bins.
        height            : int - Total figure height in pixels.
        show              : bool - If True, displays the figure in browser.

    Returns:
        plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "Computation Time Distribution",
            "Safety Margin h (Evolution)",
            "Trajectory Error (Evolution)",
            "Time-Scaling Factor (Evolution)",
        ),
        row_heights=[0.25, 0.25, 0.25, 0.25],
        vertical_spacing=0.12,
    )

    # Panel 1: Runtimes
    fig.add_trace(
        go.Histogram(x=computation_times, name="Total Time", opacity=0.6, nbinsx=nbins),
        row=1, col=1,
    )

    # Panel 2: Barrier function h
    fig.add_trace(
        go.Scatter(y=h_log, mode='lines+markers', name='h (Safety Margin)'),
        row=2, col=1,
    )

    # Panel 3: Trajectory error
    fig.add_trace(
        go.Scatter(y=trj_error_log, mode='lines+markers', name='Trajectory Error'),
        row=3, col=1,
    )

    # Panel 4: Time scaling s_dot
    fig.add_trace(
        go.Scatter(y=scaling_log, mode='lines+markers', name='Scaling Factor (s_dot)'),
        row=4, col=1,
    )

    fig.update_layout(
        barmode='overlay',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=60, r=20, t=80, b=60),
    )

    fig.update_xaxes(title_text="Computation Time [s]", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_yaxes(title_text="h [m]", row=2, col=1)

    fig.update_xaxes(title_text="Iteration", row=3, col=1)
    fig.update_yaxes(title_text="Error [rad]", row=3, col=1)

    fig.update_xaxes(title_text="Iteration", row=4, col=1)
    fig.update_yaxes(title_text="Scaling", row=4, col=1)

    if show:
        fig.show()
    return fig


def print_stats_table(stats: Dict[str, np.ndarray]) -> None:
    """Prints a formatted summary table of execution metrics with percentiles (ms)."""
    print(f"{'Name':<30} {'Mean':>12} {'50%':>12} {'90%':>12} {'95%':>12} {'99%':>12}")
    print("-" * 90)
    for name, data in stats.items():
        arr = np.asarray(data) * 1000.0
        mean_val = np.mean(arr)
        q50, q90, q95, q99 = np.quantile(arr, [0.50, 0.90, 0.95, 0.99])
        print(f"{name:<30} {mean_val:12.6f} {q50:12.6f} {q90:12.6f} {q95:12.6f} {q99:12.6f}")


def plot_lambdas(
    t_list: Sequence[float],
    gamma_list: Sequence[float],
    lambda_pos_list: Sequence[float],
    lambda_vel_list: Sequence[float],
    lambda_acc_list: Sequence[float],
    lambda_scaling_list: Sequence[float],
) -> None:
    """Plots the evolution of gamma and objective weights over time on synchronized subplots."""
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(t_list, gamma_list, label=r'$\gamma$', color='purple', linewidth=2)
    axs[0].set_ylabel('Gamma')
    axs[0].legend(loc='best')
    axs[0].grid(True, linestyle=':', alpha=0.7)
    axs[0].set_title('Evolution of Gamma and Objective Multipliers')

    axs[1].plot(t_list, lambda_pos_list, label=r'$\lambda_{pos}$', color='blue', linewidth=2)
    axs[1].set_ylabel('Pos')
    axs[1].legend(loc='best')
    axs[1].grid(True, linestyle=':', alpha=0.7)

    axs[2].plot(t_list, lambda_vel_list, label=r'$\lambda_{vel}$', color='orange', linewidth=2)
    axs[2].set_ylabel('Vel')
    axs[2].legend(loc='best')
    axs[2].grid(True, linestyle=':', alpha=0.7)

    axs[3].plot(t_list, lambda_acc_list, label=r'$\lambda_{acc}$', color='green', linewidth=2)
    axs[3].set_ylabel('Acc')
    axs[3].legend(loc='best')
    axs[3].grid(True, linestyle=':', alpha=0.7)

    axs[4].plot(t_list, lambda_scaling_list, label=r'$\lambda_{scaling}$', color='red', linewidth=2)
    axs[4].set_ylabel('Scaling')
    axs[4].legend(loc='best')
    axs[4].grid(True, linestyle=':', alpha=0.7)
    axs[4].set_xlabel('Time (t) [s]', fontsize=12)

    plt.tight_layout()


def compute_dynamic_risk_index(
    end_eff_pos: np.ndarray,
    end_eff_vel: np.ndarray,
    obs_positions: np.ndarray,
    obs_velocities: np.ndarray,
    obs_accelerations: np.ndarray,
    a_s: float = 2.5,
    T_r: float = 0.15,
    delta: float = 1.25,
    D_0: float = 0.25,
    lambd: float = 1.0,
) -> float:
    """
    Computes dynamic risk index (S_index) taking into account human acceleration
    and robot reaction time (T_r) within the predictive braking horizon.

    Parameters:
        end_eff_pos       : np.ndarray (3,) - End-effector TCP position.
        end_eff_vel       : np.ndarray (3,) - End-effector TCP velocity.
        obs_positions     : np.ndarray (N, 3) - Human obstacle keypoint positions.
        obs_velocities    : np.ndarray (N, 3) - Human obstacle keypoint velocities.
        obs_accelerations : np.ndarray (N, 3) - Human obstacle keypoint accelerations.
        a_s               : float - Maximum robot deceleration [m/s^2].
        T_r               : float - Robot system reaction time [s].
        delta             : float - Risk factor.
        D_0               : float - Static uncertainty / protective margin [m].
        lambd             : float - Index scaling parameter.

    Returns:
        s_index_max : float - Maximum risk index across all keypoints.
    """
    v_m_norm = np.linalg.norm(end_eff_vel)
    T_b = 0.0 if v_m_norm < 1e-5 else (v_m_norm / a_s)
    T_tot = T_r + T_b

    s_index_max = 0.0

    for i in range(obs_positions.shape[0]):
        p_o = obs_positions[i]
        v_hand = obs_velocities[i]
        a_hand = obs_accelerations[i]

        diff_ot = p_o - end_eff_pos
        dist_ot = np.linalg.norm(diff_ot)

        if dist_ot < 1e-5:
            return float('inf')

        dir_ot = diff_ot / dist_ot

        # Robot displacement (reaction distance + braking distance)
        delta_x_robot = (end_eff_vel * T_r) + (0.5 * end_eff_vel * T_b)

        # Human displacement (constant acceleration over T_tot)
        delta_x_hand = (v_hand * T_tot) + (0.5 * a_hand * (T_tot ** 2))

        # Total relative displacement projected along approach direction
        delta_x_tot = delta_x_hand + delta_x_robot
        integral_val = np.dot(delta_x_tot, dir_ot)

        # Accumulated risk threshold
        D_lh = delta * integral_val + D_0

        # Relative speed denominator
        v_sum = v_hand + end_eff_vel
        den = np.dot(v_sum, diff_ot)

        if den <= 1e-6:
            current_s_index = 0.0
        else:
            fraction = (dist_ot - D_lh) / den
            current_s_index = lambd * (T_tot + fraction * dist_ot)

        if current_s_index > s_index_max:
            s_index_max = current_s_index

    return s_index_max