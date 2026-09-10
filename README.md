# CBF Python: Control Barrier Functions & Safe Motion Planning for Manipulators

A modular, high-performance library and ROS 2 package implementing **Control Barrier Functions (CBF)**, **Speed and Separation Monitoring (SSM)**, **Piecewise Polynomial Dynamic Adaptation**, and **Stochastic/Gaussian Process Parameter Modulation** for industrial robot manipulators (Universal Robots UR10/UR10e).

---

## 1. Overview & Architecture

The repository is organized by strict separation of concerns into isolated modular packages:

```
cbf_python/
├── config/                  # YAML configurations (All scripts parameterized here, NO CLI args)
│   ├── bridges.yaml
│   ├── controller_defaults.yaml
│   ├── optimization.yaml
│   ├── run_cbf_optimal.yaml
│   ├── run_cbf_pid.yaml
│   ├── run_dynamic_polynomial.yaml
│   ├── run_gaussian_control.yaml
│   └── run_obstructive_test.yaml
├── cbf_python/
│   ├── bridges/             # Abstract & concrete robot command bridges (Hardware / Simulation)
│   ├── controllers/         # Real-time CBF task controllers & Numba computational kernels
│   │   └── kernels/         # Numba accelerated distance, Jacobian, and QP builders
│   ├── examples/            # Ready-to-run experiments, benchmarks, and optimization routines
│   ├── params_csv/          # Pre-computed parameter sets and Optuna trial logs
│   ├── skeleton_vectors/    # Human motion capture / skeleton pose sequences
│   ├── trajectory/          # Multi-joint & SE3 trapezoidal interpolation planners
│   └── utils/               # Configuration loaders, metrics, obstacle generators, visualizers
├── launch/                  # ROS 2 launch files
├── test/                    # Pytest verification test suite
├── package.xml              # ROS 2 package manifest
├── setup.py                 # Setuptools entry points and data file mappings
└── README.md
```

---

## 2. Environment Setup & Prerequisites

Before running any script or tests, activate the virtual environment and ensure the CMEEL Pinocchio libraries are resolved in `LD_LIBRARY_PATH`:

```bash
# 1. Activate Python virtual environment
source /home/galileo/projects/python_venv/galileo_venv/bin/activate

# 2. Export CMEEL shared libraries path
export LD_LIBRARY_PATH=/home/galileo/projects/python_venv/galileo_venv/lib/python3.12/site-packages/cmeel.prefix/lib:$LD_LIBRARY_PATH

# 3. Source ROS 2 Jazzy workspace
source /opt/ros/jazzy/setup.bash
source /home/galileo/projects/cbf_ws/install/setup.bash
```

> **Note**: `cbf_python/__init__.py` also automatically prioritizes the CMEEL virtualenv paths upon import.

---

## 3. Configuration & Parameterization

All scripts and controllers are parameterized via YAML configuration files located in `config/`. **No CLI argument parsing (`argparse`) is used.**

To modify experiment behavior (duration, hardware bridge vs. simulation, visualization, weight multipliers, obstacle parameters):
1. Open the relevant file in `config/` (e.g. `config/run_cbf_optimal.yaml`).
2. Adjust the desired parameter or flag.
3. Run the script.

### Configuration Files Summary:
- `config/run.yaml`: **Unified execution configuration file** containing parameter sections for all experiment runners (`cbf_optimal`, `cbf_pid`, `dynamic_polynomial`, `gaussian_control`, `obstructive_test`).
- `config/controller_defaults.yaml`: Fundamental controller properties ($T_c$, reaction time $T_r$, deceleration $a_s$, barrier margin $C$, gain $\gamma$, tube limits, joint limits).
- `config/bridges.yaml`: Hardware joint bridge parameters and fake simulation bridge skeleton file paths / camera transformations.
- `config/optimization.yaml`: Optuna multi-objective study settings, hyperparameter ranges, and scenario weights.

---

## 4. Running Experiments and Scripts

All scripts can be executed either directly with Python or via `ros2 run`:

```bash
# 1. Baseline Optimal CBF Controller
python3 -m cbf_python.examples.run_cbf_optimal
# or
ros2 run cbf_python run_cbf_optimal

# 2. Dynamic Polynomial Adaptation Controller
python3 -m cbf_python.examples.run_dynamic_polynomial

# 3. Gaussian / Stochastic Process Controller
python3 -m cbf_python.examples.run_gaussian_control

# 4. Obstructive Dynamic Obstacle Test
python3 -m cbf_python.examples.run_obstructive_test

# 5. Cartesian PID + CBF Controller
python3 -m cbf_python.examples.run_cbf_pid

# 6. Sequential Multi-scenario Benchmark
python3 -m cbf_python.examples.run_dynamic_pol_subsequent

# 7. Parameter Optimization (Optuna)
python3 -m cbf_python.examples.run_optimization
python3 -m cbf_python.examples.run_optimization_poly
python3 -m cbf_python.examples.run_optimization_gpr
python3 -m cbf_python.examples.run_optimization_obstructive

# 8. Interactive Plotting & Diagnostics
python3 -m cbf_python.examples.plot_metrics
python3 -m cbf_python.examples.debug_cost

# 9. Hardware Bridge Sine Wave Verification
python3 -m cbf_python.examples.bridge_sine_test
```

### ROS 2 Launch Files:
```bash
# Launch Optimal CBF controller with ZED tracking
ros2 launch cbf_python example_with_logging.launch.py

# Launch Cartesian PID + CBF controller with logging
ros2 launch cbf_python example_with_logging_PID.launch.py
```

---

## 5. Running the Test Suite

The package includes a comprehensive unit test suite covering kinematics, numerical gradients, Lie derivatives, QP assembly, controllers, bridges, trajectories, and config loading:

```bash
cd /home/galileo/projects/cbf_ws/cbf_python
pytest test/ -v
```

---

## 6. Comprehensive Map of All Files, Classes, and Functions

### A. Controllers & Kernels (`cbf_python/controllers/`)

#### `cbf_python/controllers/base_optimal_controller.py`
- `class ControllerConfig`: Dataclass holding control cycle $T_c$, barrier distance constants $C, T_r, a_s$, CBF gain $\gamma$, weights ($\lambda_{pos}, \lambda_{vel}, \lambda_{acc}, \lambda_{scaling}$), joint and tube limits.
- `class BCFOptimalController`:
  - `__init__(model_wrapper, cfg, useCbf, keypoint_to_log)`: Initializes Pinocchio model, Numba integrator blocks, and QP matrices.
  - `reset_state(q0, dq0, trajectory_time, Dtrajectory_time)`: Resets controller state and buffers.
  - `step(obs_pos, obs_vel, obs_acc, nominal_q, nominal_Dq, nominal_DDq, ref_scaling)`: Assembles QP with CBF constraints, solves with Quadprog, applies unfeasible fallback damping, and integrates dynamics.
  - `update_parameters(h, d, v_rel)`: Virtual hook for derived adaptive controllers.

#### `cbf_python/controllers/polynomial_controller.py`
- `compute_generic_lambda(h, ht, params)`: Evaluates the piecewise polynomial transition function $\lambda(h)$.
- `class PolynomialControllerConfig(ControllerConfig)`: Holds boundary parameters ($\lambda_0, \lambda_f$) and shape exponents ($n, m, w$) for each category (`pos`, `vel`, `acc`, `scaling`, `gamma`).
  - `generate_poly_dict()`: Packages parameters into structured lookup arrays.
  - `check_config_integrity()`: Validates mathematical non-negativity and absence of discontinuous jumps.
  - `normalize_parameters()`: Ensures exponent ordering $n \le m$.
- `class PolynomialOptimalController(BCFOptimalController)`:
  - `update_parameters(h, d, v_rel)`: Continuously modulates QP cost weights according to safety margin $h$.

#### `cbf_python/controllers/gaussian_controller.py`
- `class GaussianSet`: Encapsulates a reference state Gaussian distribution (means, covariance, reference $\lambda$ parameters).
- `class GaussianControllerConfig(ControllerConfig)`: Manages multiple Gaussian evaluation sets.
  - `precompute_gaussian_parameters()`: Precomputes inverse covariances and normalization factors.
- `class GaussianController(BCFOptimalController)`:
  - `update_parameters(h, d, v_rel)`: Modulates QP cost weights via multivariate Mahalanobis distance weighting across Gaussian sets.

#### `cbf_python/controllers/pid_cbf_controller.py`
- `class UR10CBFController`:
  - `__init__(model, tool_frame_name, frames_ids, Tc, Kp_tra, Kd_tra, Kp_rot, Kd_rot, gamma, useCbf, ...)`: Cartesian PID + CBF QP controller.
  - `reset_state(q0, dq0)`: Resets state.
  - `matrix_ensemble(J, dJ, dq, dtwist_tool)`: Assembles Cartesian tracking quadratic cost $P, b$.
  - `step(goal_pose, twist_goal, obstacle_positions, obstacle_velocities, obstacle_accelerations, ...)`: Computes Cartesian PID acceleration, assembles CBF constraints, solves QP for $\ddot{q}$, and integrates.

#### `cbf_python/controllers/velocity_scaling.py`
- `compute_velocity_scaling_for_human_proximity(...)`: Computes speed and separation monitoring (SSM) trajectory scaling factor based on ISO/TS 15066 standards.

#### `cbf_python/controllers/kernels/ssm_cbf_acc.py` (Numba Accelerated)
- `dmin_and_jacobian_numba(d, v_r, v_h, a_h, tr, a_max, atol)`: Analytic minimum braking distance and partial derivatives w.r.t. $[d, v_r, v_h, a_h]$.
- `h_and_jacobian_numba(...)`: Barrier function $h$ and gradient $\nabla_\psi h$.
- `jacobian_psi_times_fg_fast_numba(...)`: Computes state-space drift and input mapping Lie products $J_\psi f$ and $J_\psi g$.
- `compute_h_and_lie_numba(...)`: Evaluates $h$, Lie derivative $L_f h$, and input Lie vector $L_g h$.
- `compute_h_and_constraints_numba(...)`: Fully assembles CBF linear inequality row and upper bound.

#### `cbf_python/controllers/kernels/numba_kernels.py` (Numba Accelerated)
- `build_free_forced_one_step(Tc, nq)`: Discretized single-step linear state transition matrices.
- `fill_scaling_rows(A, c, row, nq, Tc, Dtraj, DDtraj_max)`: Trajectory scaling limits.
- `fill_tube_rows(...)`: Invariant tube position deviation constraints.
- `fill_vel_rows(...)`: Joint velocity limits.
- `fill_acc_rows(...)`: Joint acceleration limits.
- `append_cbf_rows_loop(...)`: Loops over all robot frames and obstacles to append active CBF constraints.
- `assemble_objective_parts_inplace(...)`: Assembles tracking and scaling quadratic cost terms.
- `assemble_qp_inplace(...)`: Complete QP matrix and constraint assembly kernel.

#### `cbf_python/controllers/kernels/cbf_numba_lib.py` (Numba Accelerated)
- `assemble_qp_PID_problem(...)`: Assembles joint limits, acceleration limits, and CBF constraints for Cartesian PID controller.
- `compute_q_ref_from_goal(...)`: Inverse kinematics solver for Cartesian reference pose.
- `damped_pinv_svd(J, damping)`: SVD-based damped pseudo-inverse for singularity handling.

---

### B. Robot Command Bridges (`cbf_python/bridges/`)

#### `cbf_python/bridges/base_bridge.py`
- `class BaseCommandBridgeABC(ABC)`: Abstract base class for all hardware and simulation bridges.
  - `sendCommand(q)`: Validates safety threshold jump limits before transmission.
  - `getPositions()`, `getVelocities()`, `getAccelerations()`: Thread-safe joint state getters.
  - `getObstacles(elapsed)`: Abstract method for obstacle positions, velocities, accelerations.
  - `shutdown()`: Closes communication channels.

#### `cbf_python/bridges/fake_bridge.py`
- `class FakeCommandBridge(BaseCommandBridgeABC)`: Simulated bridge replaying recorded human mo-cap skeleton CSV data.
  - `sendCommand(q)`: Updates internal simulated joint state.
  - `getObstacles(elapsed)`: Interpolates human obstacle keypoints at time `elapsed`.

#### `cbf_python/bridges/human_pose_reader.py`
- `class PoseReader`: Loads, transforms, and interpolates human keypoint trajectories from CSV files.
  - `getPose(time_val)`: Interpolated keypoint positions.
  - `getVelocity(time_val)`: Interpolated keypoint velocities.
  - `getAcceleration(time_val)`: Interpolated keypoint accelerations.

#### `cbf_python/bridges/joint_bridge.py`
- `class JointStateCommandBridge(BaseCommandBridgeABC)`: ROS 2 Humble/Jazzy bridge communicating with UR robot driver and ZED camera tracking.
  - `wait_for_first_state(target_name, timeout)`: Blocks until valid robot joint telemetry is received.
  - `switch_to_forward_position_controller_service()`: Calls ROS 2 controller manager to activate forward position control.
  - `_joint_states_callback(msg)`: Synchronous subscriber for robot joint states.
  - `_objects_callback(msg)`: Subscriber for ZED camera obstacle keypoints.

---

### C. Trajectory Planning (`cbf_python/trajectory/`)

#### `cbf_python/trajectory/trapezoid.py`
- `trapezoid_coeffs(dist, vmax, amax)`: Computes acceleration, constant-velocity, peak velocity, and total time for 1-D trapezoids.
- `scalar_trap_unit_progress(t, t_acc, t_const, v_peak, amax, t_target)`: Evaluates normalized unit progress $s(t) \in [0, 1]$, $\dot{s}(t)$, $\ddot{s}(t)$.

#### `cbf_python/trajectory/joint_interpolator.py`
- `class SegmentedJointTrap`: Multi-joint piecewise trapezoidal trajectory generator.
  - `addWayPoint(q)`: Adds a multi-joint configuration waypoint.
  - `computeTime()`: Calculates total execution time across all joints.
  - `getMotionLaw(t)`: Returns $(q(t), \dot{q}(t), \ddot{q}(t))$ at time $t$.

#### `cbf_python/trajectory/se3_interpolator.py`
- `class SegmentedSE3Trap`: Cartesian SE3 waypoint trajectory generator with analytic spatial twist and acceleration.
  - `addWayPoint(T)`: Adds an SE3 pose waypoint.
  - `computeTime()`: Computes total trajectory time.
  - `getMotionLaw(t)`: Returns $(T(t), \mathcal{V}(t), \dot{\mathcal{V}}(t))$ (pose, spatial twist, spatial acceleration).

---

### D. Utilities (`cbf_python/utils/`)

#### `cbf_python/utils/config_loader.py`
- `get_package_root()`, `get_config_path(name)`: Resolves filesystem locations.
- `resolve_path(path)`: Robustly resolves relative data/config paths.
- `load_yaml(config_file, section=None)`: Parses YAML files into dictionaries with optional section filtering.
- `load_run_config(experiment_name, config_file='run.yaml')`: Loads experiment-specific settings from unified `run.yaml`.
- `populate_controller_config(cfg, data)`: Populates `ControllerConfig`, `PolynomialControllerConfig`, or `GaussianControllerConfig`.

#### `cbf_python/utils/simulation_helpers.py`
- `UR10E_JOINTS`, `HOME`, `Q10..Q40`, `WAYPOINTS`: Standard kinematic waypoints for UR10e.
- `compute_ee_pose(q, model, data, ee_frame_id)`: Forward kinematics for end-effector.
- `compute_cartesian_poses(q, model)`: End-effector poses for all standard waypoints.
- `plan_path(planner, q_start)`: Plans standard 10-waypoint cyclical trajectory.
- `bring_robot_home(cfg, q_start, home, bridge, ctrl)`: Smoothly homes the robot using trapezoidal interpolation.

#### `cbf_python/utils/metrics.py`
- `class StatisticsCalculator`: Accumulates cycle computation times, timeouts, QP infeasibility, safety violations ($h < 0$), time scaling factor, trajectory tracking error, and Cartesian Total Variation (TV).
  - `update(...)`: Per-cycle metric accumulation.
  - `calculate_stats()`: Computes summary statistics dictionary.
- `compute_dynamic_risk_index(...)`: Evaluates dynamic risk index $S_{index}$ taking into account robot reaction time and human acceleration.
- `print_stats_table(stats)`: Formats and displays timing percentiles (mean, 50%, 90%, 95%, 99%).

#### `cbf_python/utils/obstacle_generators.py`
- `generate_velocity(...)`: Calculates 3D velocity vector towards a target.
- `generate_pos_sphere(...)`: Generates random obstacle coordinates within the lower hemisphere cone of motion.
- `generate_obs_state_h_fixed(...)`: Generates synthetic dynamic obstacles that challenge the barrier boundary.

#### `cbf_python/utils/publishers.py`
- `JointTargetPublisher`, `DoubleArrayPublisher`, `TestStartPublisher`: Thread-safe ROS 2 topic publishers.
- `JointTargetCsvPublisher`, `DoubleArrayCsvPublisher`, `TestStartCsvPublisher`: Asynchronous background CSV loggers.
- `swap_csv(csv_in, csv_out, ...)`: Modifies skeleton vector CSV streams.

#### `cbf_python/utils/optimization_helpers.py`
- `save_data(study, filename)`: Saves best Optuna trials.
- `save_data_multiobj(study, filename, weights)`: Multi-objective Min-Max normalized cost ranking and logging.
- `save_data_multitrial(study, filename, weights, scenarios)`: Multi-scenario Euclidean cost ranking.
- `run_episode_with_timeout(fn, timeout, ...)`: Subprocess wrapper with hard timeout killing divergent trials.
- `import_optuna_csv(...)`, `read_config_data_from_csv(...)`, `read_poly_config_data_from_csv(...)`: Parses stored trial parameters into controller configurations.

#### `cbf_python/utils/visualizer.py`
- `class VisualizationDaemon`: Background 60 Hz thread rendering robot pose, moving obstacle spheres, velocity vectors, HUD metrics, and trajectories in Meshcat.
- `class StochasticCBFVisualizer`: Collects stochastic barrier state vectors and computes empirical mean and covariance matrices.
- `make_summary_figure(times, h, err, scale)`: Generates interactive 4-panel Plotly diagnostic figure.
- `plot_lambdas(t, gamma, l_pos, l_vel, l_acc, l_scale)`: 5-panel Matplotlib plot of parameter evolution.
- `launch_interactive_polynomial_viewer()`: GUI with sliders for tuning polynomial adaptation curves.

#### `cbf_python/utils/limits.py`
- `make_joint_limits(nv)`: Standard joint velocity and acceleration bounds.
- `cartesian_limits_at_q(model, data, q, frame_id, Dq_max, DDq_max)`: Cartesian velocity and acceleration norms at configuration $q$.
- `sample_cartesian_limits(model, frame_id, n_samples)`: Monte Carlo sampling of maximum operational Cartesian limits.

---

### E. Executable Scripts (`cbf_python/examples/`)

- `run_cbf_optimal.py`: Main executable for the baseline optimal CBF QP controller.
- `run_dynamic_polynomial.py`: Main executable for continuous polynomial weight adaptation.
- `run_gaussian_control.py`: Main executable for stochastic / Gaussian Process CBF control.
- `run_obstructive_test.py`: Main executable for dynamic obstacle injection and evasion testing.
- `run_cbf_pid.py`: Main executable for Cartesian PID + CBF control.
- `run_dynamic_pol_subsequent.py`: Benchmarks multiple controller types in sequence.
- `run_optimization.py`: Optuna multi-objective optimization for baseline controller parameters.
- `run_optimization_poly.py`: Optuna optimization for polynomial parameters ($l_0, l_f, n, m, w$).
- `run_optimization_gpr.py`: Optuna study for stochastic / Gaussian controller tuning.
- `run_optimization_obstructive.py`: Optuna study for obstructive obstacle scenarios.
- `plot_metrics.py`: Visualizes comparative controller metrics from CSV datasets.
- `rebuild_dataset.py`: Re-evaluates and exports best trials from Optuna PostgreSQL/SQLite database.
- `debug_cost.py`: Interactive script for fine-tuning multi-objective cost weight formulations.
- `bridge_sine_test.py`: Hardware bridge verification script driving wrist 3 with a sinusoidal motion.

---

## 7. License & Citation

This software is distributed under the Apache-2.0 License.
