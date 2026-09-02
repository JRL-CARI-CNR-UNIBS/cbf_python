# CBF Python: Optimal Bounded Control Barrier Function (B-CBF) Controller

This repository provides the official implementation of the **Optimal Bounded Control Barrier Function (B-CBF)** controller for robotic manipulators operating in collaborative human-robot environments.

The framework enforces ISO/TS 15066 compliant **Speed and Separation Monitoring (SSM)** safety guarantees while optimally tracking a nominal joint trajectory via dynamic time parameterization (trajectory time scaling) and Cartesian bounding tube constraints.

---

## Key Features

- **Speed and Separation Monitoring (SSM) B-CBF Formulation**: Closed-form, analytical Control Barrier Functions considering instantaneous robot and human velocities, human maximum acceleration, and robot braking reaction dynamics.
- **Real-Time Quadratic Programming (QP)**: Solves a per-cycle QP at $500\,\text{Hz} - 1\,\text{kHz}$ computing optimal joint accelerations $\ddot{q}$ and trajectory time scaling acceleration $\ddot{s}$.
- **Bounding Tube Constraints**: Constrains robot joint deviations to dynamically safe tubes around the reference trajectory.
- **Fail-Safe Fallback & Recovery**: Automatic emergency deceleration fallback under infeasibility and seamless post-disturbance tube recovery.
- **Numba Just-In-Time (JIT) Acceleration**: Fast parallel evaluation of Lie derivatives ($L_f h, L_g h$) and Jacobian projections.
- **Modular Hardware & Perception Bridges**: Unified abstract command interface supporting both live ROS 2 robot nodes and offline dataset replay.

---

## Repository Structure

```text
.
├── config/
│   └── optimal_cbf_params.yaml         # Central YAML configuration file
├── Controller/
│   ├── optimal_cbf_task_controller.py  # BCFOptimalController & ControllerConfig
│   ├── compute_velocity_scaling_for_human_proximity.py
│   └── Numba_scripts/
│       ├── ssm_cbf_acc.py              # Analytical SSM-CBF Lie derivatives & kernels
│       └── numba_kernels.py            # Fast QP matrix & constraint assembly
├── Command_bridge/
│   ├── base_command_bridge_abc.py      # Abstract hardware/perception bridge interface
│   ├── joint_command_bridge.py         # ROS 2 live node implementation
│   ├── fake_command_bridge.py          # Offline perception replay simulation bridge
│   └── human_pose_reader.py            # CSV human skeleton coordinate reader
└── scripts/
    ├── example_cbf_optimal.py          # Main executable simulation/control script
    ├── test/
    │   └── cbf_test.py                 # Pytest validation test suite
    └── util/
        ├── joint_interpolator.py       # Multi-waypoint trapezoidal trajectory generator
        ├── visualization_daemon.py     # Background non-blocking Meshcat visualizer
        ├── statistics_calculator.py    # Safety, tracking, and efficiency metrics
        ├── bcf_utils.py                # ISO dynamic risk index computations
        └── test_utils.py               # Waypoints and kinematic helper utilities
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) (Rigid-body dynamics library)
- `quadprog` (Fast Quadratic Programming solver)
- `numba`, `numpy`, `scipy`, `pyyaml`, `meshcat`

### Environment Activation
```bash
# Activate your python virtual environment:
source /home/galileo/projects/python_venv/galileo_venv/bin/activate

# Add the workspace root to PYTHONPATH:
export PYTHONPATH=/path/to/cbf_python:$PYTHONPATH
```

---

## Tutorial: Initializing and Using the Controller

### 1. Configuration & Parameters

The controller parameters are defined in [`config/optimal_cbf_params.yaml`](config/optimal_cbf_params.yaml) or configured programmatically via [`ControllerConfig`](Controller/optimal_cbf_task_controller.py):

```python
import numpy as np
from Controller.optimal_cbf_task_controller import ControllerConfig

# 1. Initialize configuration with the loop period (e.g. 500 Hz -> 2 ms)
cfg = ControllerConfig(Tc=0.002)

# 2. Configure safety & SSM parameters
cfg.gamma = 5.95           # CBF decay gain (alpha(h) = gamma * h)
cfg.Tr = 0.15              # Reaction time buffer [s]
cfg.a_s = 2.5              # Maximum robot braking deceleration [m/s^2]
cfg.C = 0.25               # Protective separation distance [m]
cfg.max_obstacles = 90     # Maximum monitored human keypoints

# 3. Configure optimization cost weights
cfg.lambda_pos = 2098.0    # Position tracking error weight
cfg.lambda_vel = 0.343     # Velocity tracking error weight
cfg.lambda_scaling = 16.56 # Trajectory scaling regularization weight (encourages s_dot -> 1)
cfg.lambda_acc = 1.45e-10  # Joint acceleration regularization weight

# 4. Configure bounding tube joint deviation limits (radians)
delta_deg = 4.5
cfg.delta_q_max[0:2] = np.deg2rad(np.ones(2) * delta_deg * 1.0)
cfg.delta_q_max[2:4] = np.deg2rad(np.ones(2) * delta_deg * 2.0)
cfg.delta_q_max[4:6] = np.deg2rad(np.ones(2) * delta_deg * 4.0)

# 5. Set robot kinematic frame identifiers
cfg.prefix = "ur10e_"
cfg.tool_frame = "ur10e_wrist_3_joint"
cfg.elbow_frame = "ur10e_upper_arm_link"
cfg.shoulder_frame = "ur10e_shoulder_link"
```

### 2. Instantiating the Controller

Pass a robot model wrapper (containing `model`, `collision_model`, and `visual_model`) to instantiate `BCFOptimalController`:

```python
from sharework import loadSharework
from Controller.optimal_cbf_task_controller import BCFOptimalController

# Load Pinocchio robot model
UR10E_JOINTS = [
    "ur10e_shoulder_pan_joint", "ur10e_shoulder_lift_joint", "ur10e_elbow_joint",
    "ur10e_wrist_1_joint", "ur10e_wrist_2_joint", "ur10e_wrist_3_joint"
]
model_wrapper = loadSharework(UR10E_JOINTS)

# Create the B-CBF Optimal Controller
controller = BCFOptimalController(
    model_wrapper=model_wrapper,
    cfg=cfg,
    useCbf=True,           # Enable CBF obstacle avoidance constraints
    keypoint_to_log=-1,    # Monitor all obstacle keypoints
)
```

### 3. Real-Time Control Loop Step

At each control cycle ($T_c$ seconds), call `controller.step(...)` with current human obstacle measurements and the nominal trajectory reference:

```python
# Initial state reset
q_init = np.array([90.0, -140.0, 140.0, -90.0, 90.0, 0.0]) * np.pi / 180.0
controller.reset_state(q_init)

# Control loop
while True:
    # 1. Query human obstacle states (positions, velocities, accelerations in Cartesian world frame)
    # Shapes: obs_pos (N, 3), obs_vel (N, 3), obs_acc (N, 3)
    obs_pos, obs_vel, obs_acc = bridge.getObstacles()

    # 2. Query nominal motion law at current scaled trajectory time s
    # Returns desired joint positions, velocities, accelerations
    nominal_q, nominal_Dq, nominal_DDq = planner.getMotionLaw(trajectory_time)

    # 3. Execute controller optimization step
    out = controller.step(
        obs_pos=obs_pos,
        obs_vel=obs_vel,
        obs_acc=obs_acc,
        nominal_q=nominal_q,
        nominal_Dq=nominal_Dq,
        nominal_DDq=nominal_DDq,
    )

    # 4. Extract control outputs
    q_cmd = out["q"]                       # Integrated joint positions to send to robot actuators (6,)
    dq_cmd = out["dq"]                     # Integrated joint velocities (6,)
    ddq_opt = out["ddq"]                   # Optimal joint accelerations (6,)
    trajectory_time = out["trajectory_time"] # Updated virtual trajectory time s
    scaling_speed = out["Dtrajectory_time"]  # Current trajectory velocity scaling factor s_dot in [0, 1]
    h_min = out["h_min"]                   # Minimum barrier value across all monitored links
    d_min = out["d_min"]                   # Minimum physical distance to human obstacles [m]
    ctrl_state = out["unfeasible_cnt"]     # Status string: "FEASIBLE", "RECOVERING", or "UNFEASIBLE"

    # 5. Dispatch command to robot bridge
    bridge.sendCommand(q_cmd)
```

---

## Complete Minimal Example

Below is a self-contained minimal script running the controller in simulation:

```python
#!/usr/bin/env python3
import time
import numpy as np
from sharework import loadSharework
from Controller.optimal_cbf_task_controller import BCFOptimalController, ControllerConfig
from scripts.util.joint_interpolator import SegmentedJointTrap

# 1. Setup Model & Configuration
joint_names = [
    "ur10e_shoulder_pan_joint", "ur10e_shoulder_lift_joint", "ur10e_elbow_joint",
    "ur10e_wrist_1_joint", "ur10e_wrist_2_joint", "ur10e_wrist_3_joint"
]
model_wrapper = loadSharework(joint_names)
cfg = ControllerConfig(Tc=0.002)

# 2. Instantiate Controller
ctrl = BCFOptimalController(model_wrapper=model_wrapper, cfg=cfg, useCbf=True)

# 3. Create Reference Trajectory
home = np.deg2rad([90.0, -140.0, 140.0, -90.0, 90.0, 0.0])
target = np.deg2rad([31.0, -78.0, 115.0, -127.0, 86.0, -32.0])

planner = SegmentedJointTrap(Dq_max=cfg.Dq_max * 0.25, DDq_max=cfg.DDq_max * 0.125)
planner.addWayPoint(home)
planner.addWayPoint(target)
planner.addWayPoint(home)
T_total = planner.computeTime()

# 4. Simulation Execution Loop
ctrl.reset_state(home)
t_sim = 0.0
trajectory_time = 0.0

# Simulated obstacle placed near the robot trajectory
obs_pos = np.array([[0.5, 0.2, 0.4]])
obs_vel = np.array([[0.0, 0.0, 0.0]])
obs_acc = np.array([[0.0, 0.0, 0.0]])

print(f"Running simulation for 5.0 seconds (trajectory period: {T_total:.2f} s)...")
while t_sim < 5.0:
    loop_start = time.perf_counter()

    nom_q, nom_dq, nom_ddq = planner.getMotionLaw(trajectory_time % T_total)

    out = ctrl.step(
        obs_pos=obs_pos,
        obs_vel=obs_vel,
        obs_acc=obs_acc,
        nominal_q=nom_q,
        nominal_Dq=nom_dq,
        nominal_DDq=nom_ddq,
    )

    q = out["q"]
    trajectory_time = out["trajectory_time"]
    t_sim += cfg.Tc

    # Synchronize real-time loop period
    elapsed = time.perf_counter() - loop_start
    if cfg.Tc > elapsed:
        time.sleep(cfg.Tc - elapsed)

print(f"Done! Final joint position: {np.round(np.rad2deg(q), 1)} deg | Min barrier h: {out['h_min']:.3f}")
```

---

## Running the Verification Tests & Examples

### Run Automated Unit Tests
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest scripts/test/cbf_test.py
```

### Run Full Simulation with Meshcat Visualization
```bash
# Using default YAML parameters:
python3 -m scripts.example_cbf_optimal

# Using a custom YAML config file:
python3 -m scripts.example_cbf_optimal --config path/to/custom_params.yaml
```

---

## Citation

If you use this work in an academic publication, please cite the associated article:
```bibtex
@article{optimal_cbf_controller_2026,
  title   = {Optimal Bounded Control Barrier Functions for Safe Human-Robot Collaboration},
  journal = {IEEE Robotics and Automation Letters / Transactions on Robotics},
  year    = {2026}
}
```