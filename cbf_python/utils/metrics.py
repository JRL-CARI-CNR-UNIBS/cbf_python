"""Performance statistics and risk index calculation utilities."""

from typing import Dict, Any, List, Optional
import numpy as np


class StatisticsCalculator:
    """Calculates and aggregates performance statistics for robot CBF controller evaluations."""

    def __init__(
        self,
        n_wp: int,
        T_total: float,
        cartesian_configs: Dict[str, Any],
        Tc: float,
        scaling_threshold: float = 0.5,
    ) -> None:
        self.n_wp = n_wp
        self.T_total = T_total
        self.cartesian_configs = cartesian_configs
        self.Tc = Tc
        self.scaling_threshold = scaling_threshold

        self.cycles = 0
        self.timeout_cycles = 0
        self.unfeasible_cnt = 0
        self.violations = 0
        self.sum_scale = 0.0
        self.trajectory_error_sum = 0.0
        self.trajectory_cart_error_sum = 0.0
        self.low_scale_count = 0
        self.on_target_count = 0
        self.lap_count = 0

        self.trj_error_log: List[float] = []
        self.traj_cart_error_log: List[float] = []
        self.s_index_log: List[float] = []
        self.computation_times: List[float] = []

        self.prec_target = -1
        self.enable_lap_count = True
        self.final_trajectory_time = 0.0

    def update(
        self,
        out: Dict[str, Any],
        trajectory_cart_err: float,
        s_index: Optional[float],
        elapsed_time: float,
        unfeasible_string: str,
        end_eff_pos: np.ndarray,
    ) -> None:
        if self.cycles == 0:
            self.cycles += 1
            return

        self.cycles += 1

        if unfeasible_string != "FEASIBLE":
            self.unfeasible_cnt += 1
        if elapsed_time > self.Tc:
            self.timeout_cycles += 1
        self.computation_times.append(elapsed_time)

        if out.get("h_min", 0.0) < 0.0 and out.get("vr_min", 0.0) < -1e-3:
            self.violations += 1

        scaling = float(out.get("Dtrajectory_time", 1.0))
        self.sum_scale += scaling
        self.trajectory_error_sum += float(out.get("trajectory_error", 0.0))
        self.trajectory_cart_error_sum += trajectory_cart_err
        if scaling < self.scaling_threshold:
            self.low_scale_count += 1

        self.trj_error_log.append(float(out.get("trajectory_error", 0.0)))
        self.traj_cart_error_log.append(trajectory_cart_err)
        if s_index is not None:
            self.s_index_log.append(s_index)

        trajectory_time = float(out.get("trajectory_time", 0.0))
        if (trajectory_time % self.T_total) < self.Tc:
            if self.enable_lap_count:
                self.lap_count += 1
                self.prec_target = -1
                self.enable_lap_count = False
        else:
            self.enable_lap_count = True
        self.final_trajectory_time = trajectory_time

        for i, q_wp in enumerate(self.cartesian_configs.values()):
            if np.linalg.norm(np.asarray(q_wp) - end_eff_pos) < 2e-3 and self.prec_target != i:
                self.on_target_count += 1
                self.prec_target = i
                break

    def calculate_stats(self) -> Dict[str, float]:
        if self.cycles < 2:
            return {}

        completed_laps = self.lap_count + ((self.final_trajectory_time % self.T_total) / self.T_total)

        on_target_rate = self.on_target_count / (self.n_wp * completed_laps) if completed_laps > 0 else 0.0
        viol_rate = self.violations / self.cycles
        mean_scale = self.sum_scale / self.cycles
        mean_trajectory_error = self.trajectory_error_sum / self.cycles
        mean_cartesian_error = self.trajectory_cart_error_sum / self.cycles
        low_scale_rate = self.low_scale_count / self.cycles
        mean_risk_index = float(np.mean(self.s_index_log)) if self.s_index_log else 0.0

        mean_tv_error = (
            float(np.sum(np.abs(np.diff(self.trj_error_log))) / self.cycles)
            if len(self.trj_error_log) > 1
            else 0.0
        )
        mean_tv_cartesian = (
            float(np.sum(np.abs(np.diff(self.traj_cart_error_log))) / self.cycles)
            if len(self.traj_cart_error_log) > 1
            else 0.0
        )

        return {
            "timeout_percentage": 100.0 * self.timeout_cycles / self.cycles,
            "avg_computation_time": float(np.mean(self.computation_times)) if self.computation_times else 0.0,
            "unfeasible_percentage": 100.0 * self.unfeasible_cnt / self.cycles,
            "lap_count": completed_laps,
            "on_target_rate": on_target_rate * 100.0,
            "violation_rate": viol_rate * 100.0,
            "mean_scaling": mean_scale,
            "mean_trajectory_error": mean_trajectory_error,
            "low_scale_rate": low_scale_rate * 100.0,
            "mean_cartesian_error": mean_cartesian_error,
            "mean_tv_joint_error": mean_tv_error * 1000.0,
            "mean_tv_cartesian_error": mean_tv_cartesian * 1000.0,
            "mean_risk_index": mean_risk_index,
        }

    def __str__(self) -> str:
        stats = self.calculate_stats()
        if not stats:
            return "Statistics calculation requires more data."

        return (
            f"timeout cycles = {self.timeout_cycles} over {self.cycles}, "
            f"percentage = {stats['timeout_percentage']:.2f}%, "
            f"average = {stats['avg_computation_time']:.5f}s\n"
            f"unfeasible cycles = {self.unfeasible_cnt} over {self.cycles}, "
            f"percentage = {stats['unfeasible_percentage']:.2f}%\n"
            f"LAP COUNT: {stats['lap_count']:.2f}\n"
            f"on target count: {self.on_target_count}\n"
            f"WAYPOINTS REACHING PERCENTAGE: {stats['on_target_rate']:.2f} %\n"
            f"VIOLATION RATE: {stats['violation_rate']:.2f} %\n"
            f"MEAN SCALING: {stats['mean_scaling']:.4f}\n"
            f"MEAN TRAJECTORY ERROR: {stats['mean_trajectory_error']:.4f}\n"
            f"LOW SCALE RATE: {stats['low_scale_rate']:.2f}%\n"
            f"MEAN CARTESIAN ERROR: {stats['mean_cartesian_error']:.4f}\n"
            f"MEAN TV JOINT ERROR: {stats['mean_tv_joint_error']:.4f}\n"
            f"MEAN TV CARTESIAN ERROR: {stats['mean_tv_cartesian_error']:.4f}\n"
            f"MEAN RISK INDEX : {stats['mean_risk_index']:.4f}"
        )


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
    v_m_norm = float(np.linalg.norm(end_eff_vel))
    T_b = 0.0 if v_m_norm < 1e-5 else v_m_norm / a_s
    T_tot = T_r + T_b

    s_index_max = 0.0

    for i in range(obs_positions.shape[0]):
        p_o = obs_positions[i]
        v_hand = obs_velocities[i]
        a_hand = obs_accelerations[i]

        diff_ot = p_o - end_eff_pos
        dist_ot = float(np.linalg.norm(diff_ot))

        if dist_ot < 1e-5:
            return float("inf")

        dir_ot = diff_ot / dist_ot

        delta_x_robot = (end_eff_vel * T_r) + (0.5 * end_eff_vel * T_b)
        delta_x_hand = (v_hand * T_tot) + (0.5 * a_hand * (T_tot ** 2))
        delta_x_tot = delta_x_hand + delta_x_robot
        integral_val = float(np.dot(delta_x_tot, dir_ot))

        D_lh = delta * integral_val + D_0

        v_sum = v_hand + end_eff_vel
        den = float(np.dot(v_sum, diff_ot))

        if den <= 1e-6:
            current_s_index = 0.0
        else:
            fraction = (dist_ot - D_lh) / den
            current_s_index = lambd * (T_tot + fraction * dist_ot)

        if current_s_index > s_index_max:
            s_index_max = current_s_index

    return s_index_max


def print_stats_table(stats: Dict[str, np.ndarray]) -> None:
    print(f"{'Name':<30} {'Mean':>12} {'50%':>12} {'90%':>12} {'95%':>12} {'99%':>12}")
    print("-" * 90)
    for name, data in stats.items():
        arr = np.asarray(data) * 1000.0
        mean_val = float(np.mean(arr))
        q50, q90, q95, q99 = [float(q) for q in np.quantile(arr, [0.50, 0.90, 0.95, 0.99])]
        print(f"{name:<30} {mean_val:12.6f} {q50:12.6f} {q90:12.6f} {q95:12.6f} {q99:12.6f}")
