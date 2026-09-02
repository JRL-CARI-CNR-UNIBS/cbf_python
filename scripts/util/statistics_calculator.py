import numpy as np

class StatisticsCalculator:
    """
    A class to encapsulate the calculation of performance statistics for the robot controller test.
    """
    def __init__(self, n_wp, T_total, cartesian_configs, Tc, scaling_threshold=0.5):
        """
        Initializes the statistics calculator.

        Args:
            n_wp (int): Number of waypoints in the trajectory.
            T_total (float): Total time for one lap of the trajectory.
            cartesian_configs (dict): Dictionary of cartesian waypoint configurations.
            Tc (float): Control loop period.
            scaling_threshold (float): Threshold below which scaling is considered "low".
        """
        # Configuration
        self.n_wp = n_wp
        self.T_total = T_total
        self.cartesian_configs = cartesian_configs
        self.Tc = Tc
        self.scaling_threshold = scaling_threshold

        # Internal state counters and accumulators
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

        # Data logs for final calculations
        self.trj_error_log = []
        self.traj_cart_error_log = []
        self.s_index_log = []
        self.computation_times = []

        # Internal state for lap/target counting
        self.prec_target = -1
        self.enable_lap_count = True
        self.final_trajectory_time = 0.0

    def update(self, out, trajectory_cart_err, s_index, elapsed_time, unfeasible_string, end_eff_pos):
        """
        Updates the statistics with data from the current control cycle.
        """
        if self.cycles == 0:  # Skip first cycle for some metrics
            self.cycles += 1
            return

        self.cycles += 1

        # Unfeasible/Timeout counts
        if unfeasible_string != "FEASIBLE":
            self.unfeasible_cnt += 1
        if elapsed_time > self.Tc:
            self.timeout_cycles += 1
        self.computation_times.append(elapsed_time)

        # Safety Violations
        if out.get("h_min", 0) < 0 and out.get("vr_min", 0) < -1e-3:
            self.violations += 1

        # Accumulators for averages
        scaling = out.get("Dtrajectory_time", 1.0)
        self.sum_scale += scaling
        self.trajectory_error_sum += out.get("trajectory_error", 0)
        self.trajectory_cart_error_sum += trajectory_cart_err
        if scaling < self.scaling_threshold:
            self.low_scale_count += 1

        # Logs for post-processing (e.g., Total Variation)
        self.trj_error_log.append(out.get("trajectory_error", 0))
        self.traj_cart_error_log.append(trajectory_cart_err)
        if s_index is not None:
            self.s_index_log.append(s_index)

        # Lap counting logic
        trajectory_time = out.get("trajectory_time", 0)
        if (trajectory_time % self.T_total) < self.Tc:
            if self.enable_lap_count:
                self.lap_count += 1
                self.prec_target = -1
                self.enable_lap_count = False
        else:
            self.enable_lap_count = True
        self.final_trajectory_time = trajectory_time

        # On-target waypoint counting
        for i, q_wp in enumerate(self.cartesian_configs.values()):
            if np.linalg.norm(q_wp - end_eff_pos) < 2e-03 and self.prec_target != i:
                self.on_target_count += 1
                self.prec_target = i
                break

    def _calculate_stats(self):
        """
        Calculates the final statistics from the accumulated data.
        Returns a dictionary of the calculated stats.
        """
        if self.cycles < 2:
            return {}

        # Finalize lap count with fractional part
        completed_laps = self.lap_count + ((self.final_trajectory_time % self.T_total) / self.T_total)

        # Calculate Rates and Means
        on_target_rate = self.on_target_count / (self.n_wp * completed_laps) if completed_laps > 0 else 0
        viol_rate = self.violations / self.cycles
        mean_scale = self.sum_scale / self.cycles
        mean_trajectory_error = self.trajectory_error_sum / self.cycles
        mean_cartesian_error = self.trajectory_cart_error_sum / self.cycles
        low_scale_rate = self.low_scale_count / self.cycles
        mean_risk_index = np.mean(self.s_index_log) if self.s_index_log else 0

        # Calculate Total Variation (a measure of error oscillation)
        mean_tv_error = np.sum(np.abs(np.diff(self.trj_error_log))) / self.cycles
        mean_tv_cartesian = np.sum(np.abs(np.diff(self.traj_cart_error_log))) / self.cycles

        stats = {
            "timeout_percentage": 100.0 * self.timeout_cycles / self.cycles,
            "avg_computation_time": np.mean(self.computation_times) if self.computation_times else 0,
            "unfeasible_percentage": 100.0 * self.unfeasible_cnt / self.cycles,
            "lap_count": completed_laps,
            "on_target_rate": on_target_rate * 100.0,
            "violation_rate": viol_rate * 100.0,
            "mean_scaling": mean_scale,
            "mean_trajectory_error": mean_trajectory_error,
            "low_scale_rate": low_scale_rate * 100.0,
            "mean_cartesian_error": mean_cartesian_error,
            "mean_tv_joint_error": mean_tv_error * 1000,
            "mean_tv_cartesian_error": mean_tv_cartesian * 1000,
            "mean_risk_index": mean_risk_index,
        }
        return stats

    def __str__(self):
        """
        Returns a formatted string of the final statistics for printing.
        """
        stats = self._calculate_stats()
        if not stats:
            return "Statistics calculation requires more data."

        return (
            f"timeout cycles = {self.timeout_cycles} over {self.cycles}, "
            f"percentage = {stats['timeout_percentage']:.2f}, "
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
