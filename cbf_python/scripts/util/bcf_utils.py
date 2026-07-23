from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

def make_summary_figure(
    computation_times,
    h_log,
    trj_error_log,
    scaling_log,
    nbins: int = 100,
    height: int = 900,
    show: bool = True,
):
    """
    Build and (optionally) show the 4-panel summary figure.

    Panels:
      1) Histogram of computation times (Total, QP, Pinocchio, SSM, Other)
      2) Evolution of safety margin h
      3) Evolution of trajectory error
      4) Evolution of time-scaling factor

    Args:
        computation_times: array-like, total loop times [s]
        computation_times_qp: array-like, QP times [s]
        computation_times_pin: array-like, Pinocchio times [s]
        computation_times_ssm: array-like, SSM times [s]
        computation_times_others: array-like, residual (“others”) times [s]
        h_log: array-like, safety margin history
        trj_error_log: array-like, trajectory error history
        scaling_log: array-like, Dtrajectory_time history
        nbins: histogram bins
        height: figure height (px)
        show: whether to call fig.show()

    Returns:
        plotly.graph_objects.Figure
    """


    # 4 rows, 1 column
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "Computation Time Distribution",
            "Safety Margin h (evolution)",
            "Trajectory Error (evolution)",
            "Time-Scaling Factor (evolution)",
        ),
        row_heights=[0.25, 0.25, 0.25, 0.25],
        vertical_spacing=0.12
    )

    # --- Row 1: Histograms ---
    fig.add_trace(
        go.Histogram(x=computation_times, name="Total", opacity=0.5, nbinsx=nbins),
        row=1, col=1
    )
    # fig.add_trace(
    #     go.Histogram(x=computation_times_qp, name="QP", opacity=0.5, nbinsx=nbins),
    #     row=1, col=1
    # )
    # fig.add_trace(
    #     go.Histogram(x=computation_times_pin, name="Pinocchio", opacity=0.5, nbinsx=nbins),
    #     row=1, col=1
    # )
    # fig.add_trace(
    #     go.Histogram(x=computation_times_ssm, name="SSM", opacity=0.5, nbinsx=nbins),
    #     row=1, col=1
    # )
    # fig.add_trace(
    #     go.Histogram(x=computation_times_others, name="Other", opacity=0.5, nbinsx=nbins),
    #     row=1, col=1
    # )

    # --- Row 2: h evolution ---
    fig.add_trace(
        go.Scatter(y=h_log, mode='lines+markers', name='h'),
        row=2, col=1
    )

    # --- Row 3: trajectory error evolution ---
    fig.add_trace(
        go.Scatter(y=trj_error_log, mode='lines+markers', name='Trajectory error'),
        row=3, col=1
    )

    # --- Row 4: scaling evolution ---
    fig.add_trace(
        go.Scatter(y=scaling_log, mode='lines+markers', name='Scaling factor'),
        row=4, col=1
    )

    # Layout (unico)
    fig.update_layout(
        barmode='overlay',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=60, r=20, t=80, b=60),
    )

    # Assi / etichette
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

def print_stats_table(stats):
    # Print header
    print(f"{'Name':<30} {'Mean':>12} {'50%':>12} {'90%':>12} {'95%':>12} {'99%':>12}")
    print("-" * 90)
    # Print each row
    for name, data in stats.items():
        mean_val = np.mean(data*1000)
        q50, q90, q95, q99 = np.quantile(data*1000, [0.50, 0.90, 0.95, 0.99])
        print(f"{name:<30} {mean_val:12.6f} {q50:12.6f} {q90:12.6f} {q95:12.6f} {q99:12.6f}")

def plot_lambdas(t_list, gamma_list, lambda_pos_list, lambda_vel_list, lambda_acc_list, lambda_scaling_list):
    """
    Plots gamma and the lambda multipliers on 5 separate subplots sharing the same time axis.
    """
    # Create 5 subplots stacked vertically, sharing the X (time) axis
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    
    # 1. Plot Gamma
    axs[0].plot(t_list, gamma_list, label=r'$\gamma$', color='purple', linewidth=2)
    axs[0].set_ylabel('Gamma')
    axs[0].legend(loc='best')
    axs[0].grid(True, linestyle=':', alpha=0.7)
    axs[0].set_title('Evolution of Gamma and Lambda Multipliers')
    
    # 2. Plot Lambda Position
    axs[1].plot(t_list, lambda_pos_list, label=r'$\lambda_{pos}$', color='blue', linewidth=2)
    axs[1].set_ylabel('Pos')
    axs[1].legend(loc='best')
    axs[1].grid(True, linestyle=':', alpha=0.7)
    
    # 3. Plot Lambda Velocity
    axs[2].plot(t_list, lambda_vel_list, label=r'$\lambda_{vel}$', color='orange', linewidth=2)
    axs[2].set_ylabel('Vel')
    axs[2].legend(loc='best')
    axs[2].grid(True, linestyle=':', alpha=0.7)
    
    # 4. Plot Lambda Acceleration
    axs[3].plot(t_list, lambda_acc_list, label=r'$\lambda_{acc}$', color='green', linewidth=2)
    axs[3].set_ylabel('Acc')
    axs[3].legend(loc='best')
    axs[3].grid(True, linestyle=':', alpha=0.7)
    
    # 5. Plot Lambda Scaling
    axs[4].plot(t_list, lambda_scaling_list, label=r'$\lambda_{scaling}$', color='red', linewidth=2)
    axs[4].set_ylabel('Scaling')
    axs[4].legend(loc='best')
    axs[4].grid(True, linestyle=':', alpha=0.7)
    
    # Set the x-axis label only on the bottom subplot
    axs[4].set_xlabel('Time (t)', fontsize=12)
    
    # Adjust layout to prevent overlapping labels
    plt.tight_layout()


import numpy as np


def compute_dynamic_risk_index(end_eff_pos, end_eff_vel, obs_positions, obs_velocities, obs_accelerations,
                                            a_s=2.5, T_r=0.15, delta=1.25, D_0=0.25, lambd=1.0):
    """
    Calcola l'indice di rischio S_index valutando l'accelerazione umana costante
    e includendo il tempo di reazione del robot (T_r) nell'orizzonte predittivo.

    Parametri:
    - end_eff_pos: array (3,) con la posizione del TCP (p_t)
    - end_eff_vel: array (3,) con la velocità del TCP (v_m)
    - obs_positions: array (N, 3) con le posizioni dei keypoint umani (p_o)
    - obs_velocities: array (N, 3) con le velocità dei keypoint umani (v_hand)
    - obs_accelerations: array (N, 3) con le acc. dei keypoint umani (a_hand)
    - a_s: decelerazione massima del robot in m/s^2
    - T_r: tempo di reazione del sistema in secondi
    - delta: fattore di rischio statico
    - D_0: incertezza statica/margine minimo (C nel paper)
    - lambd: fattore di scaling dell'indice
    """

    # Calcolo del tempo di frenata Tb basato sulla velocità attuale del TCP
    v_m_norm = np.linalg.norm(end_eff_vel)
    if v_m_norm < 1e-5:
        T_b = 0.0
    else:
        T_b = v_m_norm / a_s

    # Orizzonte temporale totale
    T_tot = T_r + T_b

    s_index_max = 0.0

    for i in range(obs_positions.shape[0]):
        p_o = obs_positions[i]
        v_hand = obs_velocities[i]
        a_hand = obs_accelerations[i]

        diff_ot = p_o - end_eff_pos
        dist_ot = np.linalg.norm(diff_ot)

        # Evita singolarità
        if dist_ot < 1e-5:
            return float('inf')

        dir_ot = diff_ot / dist_ot

        # 1. Spostamento del robot (Moto rettilineo uniforme in T_r + Decelerazione in T_b)
        delta_x_robot = (end_eff_vel * T_r) + (0.5 * end_eff_vel * T_b)

        # 2. Spostamento dell'umano (Moto uniformemente accelerato su tutto T_tot)
        delta_x_hand = (v_hand * T_tot) + (0.5 * a_hand * (T_tot ** 2))

        # 3. Spostamento relativo complessivo
        delta_x_tot = delta_x_hand + delta_x_robot

        # 4. Proiezione dello spostamento sulla direzione di avvicinamento
        integral_val = np.dot(delta_x_tot, dir_ot)

        # 5. Calcolo soglia di rischio accumulata D_{l-h}
        D_lh = delta * integral_val + D_0

        # Denominatore: velocità relativa istantanea proiettata sulla distanza
        v_sum = v_hand + end_eff_vel
        den = np.dot(v_sum, diff_ot)

        if den <= 1e-6:
            current_s_index = 0.0
        else:
            fraction = (dist_ot - D_lh) / den
            # Sostituiamo T_b con T_tot per coerenza con il nuovo orizzonte esteso
            current_s_index = lambd * (T_tot + fraction * dist_ot)

        if current_s_index > s_index_max:
            s_index_max = current_s_index

    return s_index_max