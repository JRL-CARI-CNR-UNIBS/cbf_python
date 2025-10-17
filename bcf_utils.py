from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

def make_summary_figure(
    computation_times,
    computation_times_qp,
    computation_times_pin,
    computation_times_ssm,
    computation_times_others,
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
    fig.add_trace(
        go.Histogram(x=computation_times_qp, name="QP", opacity=0.5, nbinsx=nbins),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=computation_times_pin, name="Pinocchio", opacity=0.5, nbinsx=nbins),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=computation_times_ssm, name="SSM", opacity=0.5, nbinsx=nbins),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=computation_times_others, name="Other", opacity=0.5, nbinsx=nbins),
        row=1, col=1
    )

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
